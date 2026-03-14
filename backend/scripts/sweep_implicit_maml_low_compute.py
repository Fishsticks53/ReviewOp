from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

import services.implicit.config as config_module
from services.implicit.config import CHECKPOINT_DIR, MODEL_DIR, ImplicitConfig


def _import_runtime_modules():
    try:
        import services.implicit.encoder as encoder_module
        import services.implicit.episode_sampler as sampler_module
        import services.implicit.maml_eval as eval_module
        import services.implicit.maml_model as model_module
        import services.implicit.maml_train as train_module
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        print(
            "Missing dependency while loading training modules: "
            f"{missing}\n"
            "Install project dependencies first, then rerun.\n"
            "Example: pip install -r backend/requirements.txt",
            file=sys.stderr,
        )
        raise SystemExit(2) from exc

    return (
        encoder_module,
        sampler_module,
        eval_module,
        model_module,
        train_module,
    )


def _set_config_everywhere(
    new_config: ImplicitConfig,
    encoder_module,
    sampler_module,
    eval_module,
    model_module,
    train_module,
) -> None:
    config_module.CONFIG = new_config
    encoder_module.CONFIG = new_config
    sampler_module.CONFIG = new_config
    model_module.CONFIG = new_config
    train_module.CONFIG = new_config
    eval_module.CONFIG = new_config


def _build_run_config(
    base: ImplicitConfig,
    run_id: str,
    inner_lr: float,
    meta_lr: float,
    inner_steps: int,
    num_epochs: int,
    episodes_per_epoch: int,
    eval_episodes: int,
    seed: int,
) -> ImplicitConfig:
    ckpt_path = CHECKPOINT_DIR / f"implicit_maml_{run_id}.pt"
    metrics_path = MODEL_DIR / f"training_metrics_{run_id}.json"
    return replace(
        base,
        inner_lr=inner_lr,
        meta_lr=meta_lr,
        inner_steps=inner_steps,
        num_epochs=num_epochs,
        episodes_per_epoch=episodes_per_epoch,
        eval_episodes=eval_episodes,
        random_seed=seed,
        best_ckpt_path=ckpt_path,
        metrics_path=metrics_path,
    )


def _experiment_grid(profile: str) -> List[Dict[str, Any]]:
    if profile == "tiny":
        inner_lrs = [5e-3, 1e-2]
        meta_lrs = [1e-4, 2e-4]
        inner_steps = [1]
    else:
        inner_lrs = [1e-3, 5e-3, 1e-2]
        meta_lrs = [5e-5, 1e-4, 2e-4]
        inner_steps = [1, 3]

    combos = []
    for idx, (ilr, mlr, isteps) in enumerate(
        itertools.product(inner_lrs, meta_lrs, inner_steps),
        start=1,
    ):
        combos.append(
            {
                "run_id": f"run_{idx:02d}",
                "inner_lr": ilr,
                "meta_lr": mlr,
                "inner_steps": isteps,
            }
        )
    return combos


def _print_ranked_summary(results: List[Dict[str, Any]]) -> None:
    ranked = sorted(
        results,
        key=lambda r: (
            r.get("test_summary", {}).get("accuracy", -1.0),
            r.get("val_best_acc", -1.0),
        ),
        reverse=True,
    )
    print("\n=== Ranked Results (best first) ===")
    for i, row in enumerate(ranked, start=1):
        test_acc = row.get("test_summary", {}).get("accuracy", float("nan"))
        val_best = row.get("val_best_acc", float("nan"))
        print(
            f"{i:02d}. {row['run_id']} "
            f"test_acc={test_acc:.4f} val_best={val_best:.4f} "
            f"inner_lr={row['inner_lr']} meta_lr={row['meta_lr']} inner_steps={row['inner_steps']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Low-compute sequential sweep for implicit MAML.",
    )
    parser.add_argument("--profile", choices=["tiny", "small"], default="small")
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--episodes-per-epoch", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.set_defaults(freeze_backbone=True)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true")
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output json path; default is backend/models/implicit/sweep_results_*.json",
    )
    args = parser.parse_args()
    (
        encoder_module,
        sampler_module,
        eval_module,
        model_module,
        train_module,
    ) = _import_runtime_modules()

    base = config_module.CONFIG
    grid = _experiment_grid(args.profile)

    out_path = args.output or (MODEL_DIR / f"sweep_results_{int(time.time())}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Starting sweep profile={args.profile} runs={len(grid)}")
    print(
        "Budget "
        f"epochs={args.num_epochs} episodes/epoch={args.episodes_per_epoch} eval_episodes={args.eval_episodes}"
    )
    print(f"Results file: {out_path}")

    results: List[Dict[str, Any]] = []

    for run in grid:
        run_cfg = _build_run_config(
            base=base,
            run_id=run["run_id"],
            inner_lr=run["inner_lr"],
            meta_lr=run["meta_lr"],
            inner_steps=run["inner_steps"],
            num_epochs=args.num_epochs,
            episodes_per_epoch=args.episodes_per_epoch,
            eval_episodes=args.eval_episodes,
            seed=args.seed,
        )
        _set_config_everywhere(
            run_cfg,
            encoder_module,
            sampler_module,
            eval_module,
            model_module,
            train_module,
        )

        print(
            f"\n[{run['run_id']}] "
            f"inner_lr={run['inner_lr']} meta_lr={run['meta_lr']} inner_steps={run['inner_steps']}"
        )

        start = time.time()
        train_metrics = train_module.train_implicit_maml(
            device=args.device,
            freeze_backbone=args.freeze_backbone,
        )
        eval_report = eval_module.evaluate_implicit_maml(
            split="test",
            device=args.device,
            freeze_backbone=args.freeze_backbone,
            num_episodes=args.eval_episodes,
            output_path=(MODEL_DIR / f"eval_{run['run_id']}.json"),
        )
        elapsed = time.time() - start

        val_best_acc = max((m.val_acc for m in train_metrics), default=-1.0)
        run_result = {
            **run,
            "elapsed_sec": elapsed,
            "num_epochs": args.num_epochs,
            "episodes_per_epoch": args.episodes_per_epoch,
            "eval_episodes": args.eval_episodes,
            "freeze_backbone": args.freeze_backbone,
            "metrics_path": str(run_cfg.metrics_path),
            "checkpoint_path": str(run_cfg.best_ckpt_path),
            "val_best_acc": float(val_best_acc),
            "test_summary": eval_report.get("summary", {}),
        }
        results.append(run_result)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        print(
            f"[{run['run_id']}] done "
            f"test_acc={run_result['test_summary'].get('accuracy', -1.0):.4f} "
            f"val_best={run_result['val_best_acc']:.4f} "
            f"time={elapsed:.1f}s"
        )

    _print_ranked_summary(results)
    print(f"\nSaved sweep results: {out_path}")


if __name__ == "__main__":
    main()
