"""Pull arm A's training-time metrics from W&B and save a local, permanent
copy -- loss/grad-norm/reward curves exist *only* in W&B today (ART's
``backend.train()`` return value is logged straight to W&B via
``model.log()`` in ``train_grpo.py``, never written to any local file),
so this is the one genuinely at-risk piece of data from the whole run.

Usage:
    uv run python scripts/export_grpo_training_metrics.py
"""

from pathlib import Path
from typing import Any, cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mas_deepr.config import get_settings

_COLUMNS = [
    "_step",
    "loss/train",
    "loss/grad_norm",
    "loss/importance_ratio_mean",
    "loss/clipped_token_fraction",
    "loss/probs_corr",
    "train/reward",
    "train/reward_std_dev",
    "train/exception_rate",
    "train/num_sub_questions",
    "train/num_role_turns",
    "train/completion_tokens",
]

_PLOTS = [
    ("loss/train", "GRPO policy loss"),
    ("loss/grad_norm", "Gradient norm"),
    ("train/reward", "Mean reward per training step"),
]


def main() -> None:
    get_settings()  # triggers load_dotenv so WANDB_API_KEY is visible
    import wandb

    api = wandb.Api()
    run = api.run(
        "kapitanushka05-warsaw-university-of-technology/mas-deepr/mas-deepr-grpo"
    )
    print(f"run state: {run.state}")

    # wandb's Run.history() returns a pandas DataFrame at runtime; typed
    # as Any here rather than adding a pandas-stubs dependency just for
    # this one script (the rest of the codebase uses polars throughout).
    hist = cast(Any, run.history())
    present = [c for c in _COLUMNS if c in hist.columns]
    df = hist[present].sort_values("_step")

    out_dir = Path(__file__).resolve().parent.parent / "reports" / "grpo-arm-a"
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "charts").mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "data" / "training_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_path}")

    for col, title in _PLOTS:
        if col not in df.columns:
            continue
        # Each metric is logged via its own model.log() call at its own
        # step cadence (loss/* from one call, train/* from another) --
        # dropping rows where ANY column is null would wipe almost
        # everything. Select each metric's own non-null rows instead.
        sub = df[["_step", col]].dropna()
        if sub.empty:
            print(f"skipping {col}: no data")
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(sub["_step"], sub[col])
        ax.set_xlabel("training step (ART's internal backend step counter)")
        ax.set_ylabel(col)
        ax.set_title(f"mas-deepr-grpo: {title}")
        fig.tight_layout()
        fname = out_dir / "charts" / f"{col.replace('/', '_')}.png"
        fig.savefig(fname, dpi=150)
        print(f"Wrote {fname} ({len(sub)} points)")


if __name__ == "__main__":
    main()
