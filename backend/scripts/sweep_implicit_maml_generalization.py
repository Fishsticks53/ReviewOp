from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import services.implicit.config as config_module
from services.implicit.config import CHECKPOINT_DIR, MODEL_DIR, ImplicitConfig


def _import_runtime_modules():
    try:
        import torch
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        import services.implicit.encoder as encoder_module
        import services.implicit.maml_eval as eval_module
        import services.implicit.maml_model as model_module
        import services.implicit.maml_train as train_module
        from services.implicit.encoder import build_encoder, encode_episode_rows
        from services.implicit.episode_sampler import build_episode_sampler
        from services.implicit.label_maps import load_label_encoder
        from services.implicit.maml_model import build_maml_model
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(
            "Missing dependency while loading runtime modules: "
            f"{missing}\n"
            "Install project dependencies first, then rerun.\n"
            "Example: pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    return {
        "torch": torch,
        "accuracy_score": accuracy_score,
        "precision_recall_fscore_support": precision_recall_fscore_support,
        "encoder_module": encoder_module,
        "eval_module": eval_module,
        "model_module": model_module,
        "train_module": train_module,
        "build_encoder": build_encoder,
        "encode_episode_rows": encode_episode_rows,
        "build_episode_sampler": build_episode_sampler,
        "load_label_encoder": load_label_encoder,
        "build_maml_model": build_maml_model,
    }


def _set_config_everywhere(mods: Dict[str, Any], cfg: ImplicitConfig) -> None:
    config_module.CONFIG = cfg
    mods["encoder_module"].CONFIG = cfg
    mods["model_module"].CONFIG = cfg
    mods["train_module"].CONFIG = cfg
    mods["eval_module"].CONFIG = cfg


def _set_seed(mods: Dict[str, Any], seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    mods["torch"].manual_seed(seed)
    if mods["torch"].cuda.is_available():
        mods["torch"].cuda.manual_seed_all(seed)


def _build_run_config(
    base: ImplicitConfig,
    run_id: str,
    inner_lr: float,
    meta_lr: float,
    inner_steps: int,
    max_n_way: int,
    num_epochs: int,
    episodes_per_epoch: int,
    eval_episodes: int,
    seed: int,
) -> ImplicitConfig:
    return replace(
        base,
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        inner_steps=inner_steps,
        n_way=max_n_way,
        num_epochs=num_epochs,
        episodes_per_epoch=episodes_per_epoch,
        eval_episodes=eval_episodes,
        random_seed=seed,
        best_ckpt_path=CHECKPOINT_DIR / f"implicit_maml_gen_{run_id}.pt",
        metrics_path=MODEL_DIR / f"training_metrics_gen_{run_id}.json",
    )


def _grid(profile: str) -> List[Dict[str, Any]]:
    if profile == "tiny":
        inner_lrs = [5e-3, 1e-2]
        meta_lrs = [1e-4, 2e-4]
        inner_steps = [1, 3]
    else:
        inner_lrs = [1e-3, 5e-3, 1e-2]
        meta_lrs = [5e-5, 1e-4, 2e-4]
        inner_steps = [1, 3]

    rows = []
    for idx, (ilr, mlr, isteps) in enumerate(
        itertools.product(inner_lrs, meta_lrs, inner_steps),
        start=1,
    ):
        rows.append(
            {
                "run_id": f"run_{idx:02d}",
                "inner_lr": ilr,
                "meta_lr": mlr,
                "inner_steps": isteps,
            }
        )
    return rows


def _choose_n_way(choices: Sequence[int], eligible_count: int, fallback: int) -> int:
    feasible = [x for x in choices if 2 <= x <= eligible_count]
    if feasible:
        return random.choice(feasible)
    return min(max(2, fallback), eligible_count)


def _episode_forward(mods: Dict[str, Any], model, encoder, episode, device: str, training: bool):
    support_embeddings, support_labels, _ = mods["encode_episode_rows"](
        encoder=encoder,
        rows=episode.support_rows,
        text_key="evidence_sentence",
        label_key="local_label",
    )
    query_embeddings, query_labels, _ = mods["encode_episode_rows"](
        encoder=encoder,
        rows=episode.query_rows,
        text_key="evidence_sentence",
        label_key="local_label",
    )

    support_embeddings = support_embeddings.to(device)
    support_labels = support_labels.to(device)
    query_embeddings = query_embeddings.to(device)
    query_labels = query_labels.to(device)

    return model.episode_forward(
        support_embeddings=support_embeddings,
        support_labels=support_labels,
        query_embeddings=query_embeddings,
        query_labels=query_labels,
        inner_steps=config_module.CONFIG.inner_steps,
        inner_lr=config_module.CONFIG.inner_lr,
        create_graph=training,
    )


def _train_epoch_mixed_way(
    mods: Dict[str, Any],
    model,
    encoder,
    optimizer,
    device: str,
    epoch: int,
    n_way_choices: Sequence[int],
) -> Tuple[float, float]:
    model.train()
    encoder.train()
    sampler = mods["build_episode_sampler"]("train", seed=config_module.CONFIG.random_seed + epoch)

    total_loss = 0.0
    total_acc = 0.0
    steps = config_module.CONFIG.episodes_per_epoch
    eligible = len(sampler.eligible_aspects)

    for _ in range(steps):
        n_way = _choose_n_way(n_way_choices, eligible_count=eligible, fallback=config_module.CONFIG.n_way)
        episode = sampler.sample_episode(n_way=n_way)
        optimizer.zero_grad()
        out = _episode_forward(mods, model, encoder, episode, device, training=True)
        out.query_loss.backward()
        optimizer.step()
        total_loss += float(out.query_loss.item())
        total_acc += float(out.query_acc)

    return total_loss / max(1, steps), total_acc / max(1, steps)


def _eval_split_mixed_way(
    mods: Dict[str, Any],
    model,
    encoder,
    split: str,
    device: str,
    n_way_choices: Sequence[int],
    num_episodes: int,
    seed_offset: int,
) -> Dict[str, float]:
    model.eval()
    encoder.eval()
    sampler = mods["build_episode_sampler"](split, seed=config_module.CONFIG.random_seed + seed_offset)
    label_maps = mods["load_label_encoder"]()
    eligible = len(sampler.eligible_aspects)

    y_true: List[int] = []
    y_pred: List[int] = []
    total_loss = 0.0
    total_acc = 0.0

    for _ in range(num_episodes):
        n_way = _choose_n_way(n_way_choices, eligible_count=eligible, fallback=config_module.CONFIG.n_way)
        episode = sampler.sample_episode(n_way=n_way)
        out = _episode_forward(mods, model, encoder, episode, device, training=False)

        local_to_global = {
            local_id: label_maps.aspect_to_id[aspect]
            for local_id, aspect in enumerate(episode.selected_aspects)
        }
        true_global = [local_to_global[int(x)] for x in out.query_logits.argmax(dim=-1).detach().cpu().tolist()]
        pred_global = [local_to_global[int(x)] for x in out.query_preds.detach().cpu().tolist()]

        # true labels should come from query labels, re-derive from episode rows order.
        # query rows are already aligned with model output order after encode_episode_rows filtering.
        _, query_labels, _ = mods["encode_episode_rows"](
            encoder=encoder,
            rows=episode.query_rows,
            text_key="evidence_sentence",
            label_key="local_label",
        )
        true_local = query_labels.detach().cpu().tolist()
        true_global = [local_to_global[int(x)] for x in true_local]

        y_true.extend(true_global)
        y_pred.extend(pred_global)
        total_loss += float(out.query_loss.item())
        total_acc += float(out.query_acc)

    acc = float(mods["accuracy_score"](y_true, y_pred)) if y_true else 0.0
    macro_p, macro_r, macro_f1, _ = mods["precision_recall_fscore_support"](
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    ) if y_true else (0.0, 0.0, 0.0, None)

    return {
        "loss": float(total_loss / max(1, num_episodes)),
        "acc": float(total_acc / max(1, num_episodes)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "global_accuracy": acc,
    }


def _run_one(
    mods: Dict[str, Any],
    run: Dict[str, Any],
    args,
    n_way_choices: Sequence[int],
) -> Dict[str, Any]:
    max_n_way = max(n_way_choices)
    cfg = _build_run_config(
        base=config_module.CONFIG,
        run_id=run["run_id"],
        inner_lr=run["inner_lr"],
        meta_lr=run["meta_lr"],
        inner_steps=run["inner_steps"],
        max_n_way=max_n_way,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        eval_episodes=args.eval_episodes,
        seed=args.seed,
    )
    _set_config_everywhere(mods, cfg)
    _set_seed(mods, cfg.random_seed)

    device = args.device or cfg.device
    encoder = mods["build_encoder"](freeze_backbone=args.freeze_backbone, device=device)
    model = mods["build_maml_model"](
        input_dim=encoder.hidden_size,
        hidden_dim=cfg.classifier_hidden_dim,
        device=device,
    )
    optimizer = mods["train_module"].build_optimizer(model=model, encoder=encoder)

    best_f1 = -1.0
    best_epoch = 0
    wait = 0
    history: List[Dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        train_loss, train_acc = _train_epoch_mixed_way(
            mods=mods,
            model=model,
            encoder=encoder,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            n_way_choices=n_way_choices,
        )
        val = _eval_split_mixed_way(
            mods=mods,
            model=model,
            encoder=encoder,
            split="val",
            device=device,
            n_way_choices=n_way_choices,
            num_episodes=cfg.eval_episodes,
            seed_offset=1000 + epoch,
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val["loss"],
            "val_acc": val["global_accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(row)

        improved = val["macro_f1"] > best_f1
        if improved:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            wait = 0
            mods["train_module"].save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_acc=best_f1,
                path=cfg.best_ckpt_path,
            )
        else:
            wait += 1

        print(
            f"[{run['run_id']}] epoch {epoch}/{cfg.num_epochs} "
            f"train_acc={train_acc:.4f} val_acc={val['global_accuracy']:.4f} "
            f"val_f1={val['macro_f1']:.4f} wait={wait}/{args.patience}"
        )
        if wait >= args.patience:
            print(f"[{run['run_id']}] early stop at epoch {epoch}")
            break

    mods["train_module"].load_checkpoint(model=model, optimizer=None, path=cfg.best_ckpt_path, map_location=device)
    test = _eval_split_mixed_way(
        mods=mods,
        model=model,
        encoder=encoder,
        split="test",
        device=device,
        n_way_choices=n_way_choices,
        num_episodes=max(cfg.eval_episodes, args.test_eval_episodes),
        seed_offset=5000,
    )

    with cfg.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return {
        **run,
        "n_way_choices": list(n_way_choices),
        "elapsed_sec": float(time.time() - start),
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_f1),
        "best_checkpoint_path": str(cfg.best_ckpt_path),
        "train_history_path": str(cfg.metrics_path),
        "test_metrics": test,
    }


def _print_leaderboard(results: List[Dict[str, Any]]) -> None:
    ranked = sorted(
        results,
        key=lambda x: (
            x["test_metrics"]["macro_f1"],
            x["test_metrics"]["global_accuracy"],
            x["best_val_macro_f1"],
        ),
        reverse=True,
    )
    print("\n=== Leaderboard (macro-F1 first) ===")
    for i, r in enumerate(ranked, start=1):
        print(
            f"{i:02d}. {r['run_id']} test_f1={r['test_metrics']['macro_f1']:.4f} "
            f"test_acc={r['test_metrics']['global_accuracy']:.4f} "
            f"val_f1={r['best_val_macro_f1']:.4f} "
            f"inner_lr={r['inner_lr']} meta_lr={r['meta_lr']} inner_steps={r['inner_steps']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization-focused automated sweep for implicit MAML.")
    parser.add_argument("--profile", choices=["tiny", "small"], default="tiny")
    parser.add_argument("--num-epochs", type=int, default=20)
    parser.add_argument("--episodes-per-epoch", type=int, default=60)
    parser.add_argument("--eval-episodes", type=int, default=40)
    parser.add_argument("--test-eval-episodes", type=int, default=100)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-way-choices", type=str, default="2,3,4")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    n_way_choices = [int(x.strip()) for x in args.n_way_choices.split(",") if x.strip()]
    if not n_way_choices:
        raise ValueError("n-way choices cannot be empty")

    mods = _import_runtime_modules()
    base = config_module.CONFIG
    grid = _grid(args.profile)
    out_path = args.output or (MODEL_DIR / f"sweep_generalization_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting generalization sweep profile={args.profile} runs={len(grid)}")
    print(
        f"Budget epochs={args.num_epochs} episodes/epoch={args.episodes_per_epoch} "
        f"val_eval={args.eval_episodes} test_eval={args.test_eval_episodes}"
    )
    print(f"Mixed n-way choices: {n_way_choices} | patience={args.patience}")
    print(f"Output: {out_path}")

    results: List[Dict[str, Any]] = []
    for run in grid:
        print(
            f"\n[{run['run_id']}] "
            f"inner_lr={run['inner_lr']} meta_lr={run['meta_lr']} inner_steps={run['inner_steps']}"
        )
        result = _run_one(mods=mods, run=run, args=args, n_way_choices=n_way_choices)
        results.append(result)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(
            f"[{run['run_id']}] done test_f1={result['test_metrics']['macro_f1']:.4f} "
            f"test_acc={result['test_metrics']['global_accuracy']:.4f}"
        )

    _set_config_everywhere(mods, base)
    _print_leaderboard(results)
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
