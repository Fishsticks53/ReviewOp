from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
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
            f"Missing dependency: {missing}\nInstall dependencies first.",
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


def _parse_int_list(text: str) -> List[int]:
    vals = [int(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty integer list")
    return vals


def _parse_float_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("Expected non-empty float list")
    return vals


def _grid(inner_lrs: Sequence[float], meta_lrs: Sequence[float], inner_steps: Sequence[int]) -> List[Dict[str, Any]]:
    runs = []
    for idx, (ilr, mlr, isteps) in enumerate(itertools.product(inner_lrs, meta_lrs, inner_steps), start=1):
        runs.append(
            {
                "run_id": f"cfg_{idx:02d}",
                "inner_lr": float(ilr),
                "meta_lr": float(mlr),
                "inner_steps": int(isteps),
            }
        )
    return runs


def _build_cfg(base: ImplicitConfig, run_id: str, seed: int, max_n_way: int, args) -> ImplicitConfig:
    return replace(
        base,
        inner_lr=args.inner_lr,
        meta_lr=args.meta_lr,
        inner_steps=args.inner_steps,
        n_way=max_n_way,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        eval_episodes=args.eval_episodes,
        random_seed=seed,
        best_ckpt_path=CHECKPOINT_DIR / f"implicit_maml_robust_{run_id}_s{seed}.pt",
        metrics_path=MODEL_DIR / f"training_metrics_robust_{run_id}_s{seed}.json",
    )


def _choose_n_way(choices: Sequence[int], eligible_count: int, fallback: int) -> int:
    feasible = [x for x in choices if 2 <= x <= eligible_count]
    if not feasible:
        return min(max(2, fallback), eligible_count)
    return random.choice(feasible)


def _maybe_unfreeze_top_layers(encoder, n_layers: int) -> None:
    if n_layers <= 0:
        return
    backbone = getattr(encoder, "backbone", None)
    if backbone is None:
        return
    layer_list = None
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        layer_list = backbone.encoder.layer
    elif hasattr(backbone, "transformer") and hasattr(backbone.transformer, "layer"):
        layer_list = backbone.transformer.layer
    if layer_list is None:
        return
    for layer in list(layer_list)[-n_layers:]:
        for p in layer.parameters():
            p.requires_grad = True


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
    ), query_labels


def _train_epoch(mods: Dict[str, Any], model, encoder, optimizer, device: str, epoch: int, n_way_choices: Sequence[int]) -> Tuple[float, float]:
    model.train()
    encoder.train()
    sampler = mods["build_episode_sampler"]("train", seed=config_module.CONFIG.random_seed + epoch)
    steps = config_module.CONFIG.episodes_per_epoch
    eligible = len(sampler.eligible_aspects)
    total_loss = 0.0
    total_acc = 0.0
    for _ in range(steps):
        ep = sampler.sample_episode(n_way=_choose_n_way(n_way_choices, eligible, config_module.CONFIG.n_way))
        optimizer.zero_grad()
        out, _ = _episode_forward(mods, model, encoder, ep, device, training=True)
        out.query_loss.backward()
        optimizer.step()
        total_loss += float(out.query_loss.item())
        total_acc += float(out.query_acc)
    return total_loss / max(1, steps), total_acc / max(1, steps)


def _eval_split(mods: Dict[str, Any], model, encoder, split: str, device: str, n_way_choices: Sequence[int], num_episodes: int, seed_offset: int) -> Dict[str, float]:
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
        ep = sampler.sample_episode(n_way=_choose_n_way(n_way_choices, eligible, config_module.CONFIG.n_way))
        out, query_labels = _episode_forward(mods, model, encoder, ep, device, training=False)
        local_to_global = {local_id: label_maps.aspect_to_id[a] for local_id, a in enumerate(ep.selected_aspects)}
        true_local = query_labels.detach().cpu().tolist()
        pred_local = out.query_preds.detach().cpu().tolist()
        y_true.extend([local_to_global[int(x)] for x in true_local])
        y_pred.extend([local_to_global[int(x)] for x in pred_local])
        total_loss += float(out.query_loss.item())
        total_acc += float(out.query_acc)

    gacc = float(mods["accuracy_score"](y_true, y_pred)) if y_true else 0.0
    macro_p, macro_r, macro_f1, _ = mods["precision_recall_fscore_support"](
        y_true, y_pred, average="macro", zero_division=0
    ) if y_true else (0.0, 0.0, 0.0, None)
    return {
        "loss": float(total_loss / max(1, num_episodes)),
        "acc": float(total_acc / max(1, num_episodes)),
        "global_accuracy": gacc,
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
    }


def _run_one_seed(mods: Dict[str, Any], base_cfg: ImplicitConfig, run: Dict[str, Any], seed: int, args, warmup_choices: Sequence[int], full_choices: Sequence[int]) -> Dict[str, Any]:
    max_n_way = max(full_choices)
    cfg = replace(
        base_cfg,
        inner_lr=run["inner_lr"],
        meta_lr=run["meta_lr"],
        inner_steps=run["inner_steps"],
        n_way=max_n_way,
        num_epochs=args.num_epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        eval_episodes=args.eval_episodes,
        random_seed=seed,
        best_ckpt_path=CHECKPOINT_DIR / f"implicit_maml_robust_{run['run_id']}_s{seed}.pt",
        metrics_path=MODEL_DIR / f"training_metrics_robust_{run['run_id']}_s{seed}.json",
    )
    _set_config_everywhere(mods, cfg)
    _set_seed(mods, seed)

    device = args.device or cfg.device
    encoder = mods["build_encoder"](freeze_backbone=True, device=device)
    _maybe_unfreeze_top_layers(encoder, args.unfreeze_top_layers)
    model = mods["build_maml_model"](input_dim=encoder.hidden_size, hidden_dim=cfg.classifier_hidden_dim, device=device)
    optimizer = mods["train_module"].build_optimizer(model=model, encoder=encoder)

    best_f1 = -1.0
    best_epoch = 0
    wait = 0
    history: List[Dict[str, Any]] = []
    start = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        epoch_start = time.time()
        choices = warmup_choices if epoch <= args.warmup_epochs else full_choices
        tr_loss, tr_acc = _train_epoch(mods, model, encoder, optimizer, device, epoch, choices)
        val = _eval_split(mods, model, encoder, "val", device, choices, cfg.eval_episodes, 1000 + epoch)
        history.append(
            {
                "epoch": epoch,
                "train_loss": tr_loss,
                "train_acc": tr_acc,
                "val_loss": val["loss"],
                "val_acc": val["global_accuracy"],
                "val_macro_f1": val["macro_f1"],
                "n_way_choices": list(choices),
            }
        )
        if val["macro_f1"] > best_f1:
            best_f1 = val["macro_f1"]
            best_epoch = epoch
            wait = 0
            mods["train_module"].save_checkpoint(model=model, optimizer=optimizer, epoch=epoch, best_val_acc=best_f1, path=cfg.best_ckpt_path)
        else:
            wait += 1
        elapsed_epochs = time.time() - start
        avg_epoch_time = elapsed_epochs / max(1, epoch)
        remaining_epochs = max(0, cfg.num_epochs - epoch)
        eta_sec = avg_epoch_time * remaining_epochs
        print(
            f"    epoch {epoch}/{cfg.num_epochs} "
            f"val_f1={val['macro_f1']:.4f} best={best_f1:.4f} wait={wait}/{args.patience} "
            f"epoch_time={time.time()-epoch_start:.1f}s eta_seed={eta_sec/60:.1f}m"
        )
        if wait >= args.patience:
            break

    mods["train_module"].load_checkpoint(model=model, optimizer=None, path=cfg.best_ckpt_path, map_location=device)
    test = _eval_split(
        mods,
        model,
        encoder,
        "test",
        device,
        full_choices,
        max(args.test_eval_episodes, cfg.eval_episodes),
        5000,
    )
    with cfg.metrics_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

    return {
        "seed": seed,
        "elapsed_sec": float(time.time() - start),
        "best_epoch": best_epoch,
        "best_val_macro_f1": float(best_f1),
        "test_metrics": test,
        "checkpoint_path": str(cfg.best_ckpt_path),
        "train_history_path": str(cfg.metrics_path),
    }


def _aggregate_seed_runs(seed_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    f1s = [r["test_metrics"]["macro_f1"] for r in seed_runs]
    accs = [r["test_metrics"]["global_accuracy"] for r in seed_runs]
    val_f1s = [r["best_val_macro_f1"] for r in seed_runs]
    return {
        "num_seeds": len(seed_runs),
        "test_macro_f1_mean": float(statistics.mean(f1s)),
        "test_macro_f1_std": float(statistics.pstdev(f1s) if len(f1s) > 1 else 0.0),
        "test_acc_mean": float(statistics.mean(accs)),
        "test_acc_std": float(statistics.pstdev(accs) if len(accs) > 1 else 0.0),
        "val_macro_f1_mean": float(statistics.mean(val_f1s)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Robust MAML sweep: multi-seed + curriculum + partial unfreeze.")
    parser.add_argument("--inner-lrs", type=str, default="0.008,0.01,0.012")
    parser.add_argument("--meta-lrs", type=str, default="0.00015,0.0002,0.00025")
    parser.add_argument("--inner-steps", type=str, default="2,3,4")
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument("--num-epochs", type=int, default=28)
    parser.add_argument("--episodes-per-epoch", type=int, default=100)
    parser.add_argument("--eval-episodes", type=int, default=80)
    parser.add_argument("--test-eval-episodes", type=int, default=160)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--warmup-epochs", type=int, default=8)
    parser.add_argument("--warmup-n-ways", type=str, default="2,3")
    parser.add_argument("--full-n-ways", type=str, default="2,3,4")
    parser.add_argument("--unfreeze-top-layers", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    inner_lrs = _parse_float_list(args.inner_lrs)
    meta_lrs = _parse_float_list(args.meta_lrs)
    inner_steps = _parse_int_list(args.inner_steps)
    seeds = _parse_int_list(args.seeds)
    warmup_choices = _parse_int_list(args.warmup_n_ways)
    full_choices = _parse_int_list(args.full_n_ways)
    if max(warmup_choices) > max(full_choices):
        raise ValueError("warmup n-way choices must be a subset/range within full n-way choices")

    mods = _import_runtime_modules()
    base = config_module.CONFIG
    run_grid = _grid(inner_lrs, meta_lrs, inner_steps)
    out_path = args.output or (MODEL_DIR / f"sweep_robust_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: List[Dict[str, Any]] = []
    total_seed_tasks = len(run_grid) * len(seeds)
    completed_seed_tasks = 0
    completed_seed_times: List[float] = []
    for run in run_grid:
        print(
            f"\n[{run['run_id']}] inner_lr={run['inner_lr']} meta_lr={run['meta_lr']} inner_steps={run['inner_steps']}"
        )
        seed_runs = []
        for seed in seeds:
            print(f"  seed={seed} ...")
            seed_start = time.time()
            seed_result = _run_one_seed(
                mods=mods,
                base_cfg=base,
                run=run,
                seed=seed,
                args=args,
                warmup_choices=warmup_choices,
                full_choices=full_choices,
            )
            seed_runs.append(seed_result)
            seed_elapsed = time.time() - seed_start
            completed_seed_tasks += 1
            completed_seed_times.append(seed_elapsed)
            avg_seed_time = sum(completed_seed_times) / max(1, len(completed_seed_times))
            remaining_tasks = total_seed_tasks - completed_seed_tasks
            remaining_eta_sec = avg_seed_time * remaining_tasks
            print(
                f"  seed={seed} test_f1={seed_result['test_metrics']['macro_f1']:.4f} "
                f"test_acc={seed_result['test_metrics']['global_accuracy']:.4f} "
                f"time={seed_elapsed/60:.1f}m overall_eta={remaining_eta_sec/60:.1f}m "
                f"({completed_seed_tasks}/{total_seed_tasks})"
            )

        aggregate = _aggregate_seed_runs(seed_runs)
        row = {
            **run,
            "seeds": seeds,
            "warmup_n_ways": warmup_choices,
            "full_n_ways": full_choices,
            "unfreeze_top_layers": args.unfreeze_top_layers,
            "aggregate": aggregate,
            "seed_runs": seed_runs,
        }
        all_results.append(row)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)

    _set_config_everywhere(mods, base)
    ranked = sorted(
        all_results,
        key=lambda r: (r["aggregate"]["test_macro_f1_mean"], r["aggregate"]["test_acc_mean"]),
        reverse=True,
    )
    print("\n=== Robust Leaderboard (mean across seeds) ===")
    for i, r in enumerate(ranked, start=1):
        a = r["aggregate"]
        print(
            f"{i:02d}. {r['run_id']} f1={a['test_macro_f1_mean']:.4f}±{a['test_macro_f1_std']:.4f} "
            f"acc={a['test_acc_mean']:.4f}±{a['test_acc_std']:.4f} "
            f"(inner_lr={r['inner_lr']} meta_lr={r['meta_lr']} inner_steps={r['inner_steps']})"
        )
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
