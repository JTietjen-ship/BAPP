import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

HISTORY_CSV = "epoch_history.csv"
OUTPUT_DIR = "graphs"


def add_trendline(ax, x, y, color, degree: int = 2):
    """Fits and plots a smooth polynomial trendline matching the series color."""
    if len(x) <= degree:
        return  # Not enough points to fit the requested polynomial degree

    coeffs = np.polyfit(x, y, deg=degree)
    poly_fn = np.poly1d(coeffs)
    x_smooth = np.linspace(x.min(), x.max(), 200)
    ax.plot(
        x_smooth,
        poly_fn(x_smooth),
        color=color,
        linestyle="--",
        linewidth=2.2,
        alpha=0.95,
    )


def plot_training_curves(
    history_csv: str = HISTORY_CSV,
    output_dir: str = OUTPUT_DIR,
    filter_split: str | None = None,
    keep_latest_per_config: bool = True,
    trend_degree: int = 2,  # 1 = straight linear fit, 2 or 3 = curved best fit
):
    if not os.path.exists(history_csv):
        raise FileNotFoundError(f"Could not find {history_csv}. Run training first.")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(history_csv)

    if filter_split is not None:
        df = df[df["split_type"] == filter_split]
        if df.empty:
            raise ValueError(f"No records found for split_type='{filter_split}'.")

    if keep_latest_per_config:
        latest_runs = (
            df.sort_values("run_id")
            .groupby("config_label")["run_id"]
            .last()
            .values
        )
        df = df[df["run_id"].isin(latest_runs)]

    plt.style.use("seaborn-v0_8-whitegrid")
    title_suffix = f" ({filter_split} split)" if filter_split else ""
    fit_type_str = "Linear" if trend_degree == 1 else f"Poly (deg={trend_degree})"

    # ---------------------------------------------------------
    # Plot 1: Validation MSE per Epoch + Line of Best Fit
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for (run_id, label), group in df.groupby(["run_id", "config_label"], sort=False):
        group = group.sort_values("epoch")
        x = group["epoch"].to_numpy()
        y = group["val_mse"].to_numpy()

        # Plot raw points/line faintly so the trendline stands out cleanly
        (raw_line,) = ax.plot(
            x,
            y,
            marker="o",
            markersize=3.5,
            linewidth=1,
            alpha=0.35,
            label=f"{label} (dashed = {fit_type_str} fit)",
        )
        add_trendline(ax, x, y, color=raw_line.get_color(), degree=trend_degree)

    ax.set_title(
        f"Validation MSE per Epoch Across Hyperparameters{title_suffix}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Validation MSE", fontsize=11)
    ax.legend(title="Configuration", frameon=True, fontsize=8.5, title_fontsize=10, loc="best")
    fig.tight_layout()

    mse_filename = f"val_mse_per_epoch{'_' + filter_split if filter_split else ''}.png"
    mse_path = os.path.join(output_dir, mse_filename)
    fig.savefig(mse_path)
    plt.close(fig)

    # ---------------------------------------------------------
    # Plot 2: Validation Spearman Correlation per Epoch + Line of Best Fit
    # ---------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    for (run_id, label), group in df.groupby(["run_id", "config_label"], sort=False):
        group = group.sort_values("epoch")
        x = group["epoch"].to_numpy()
        y = group["val_spearman"].to_numpy()

        (raw_line,) = ax.plot(
            x,
            y,
            marker="s",
            markersize=3.5,
            linewidth=1,
            alpha=0.35,
            label=f"{label} (dashed = {fit_type_str} fit)",
        )
        add_trendline(ax, x, y, color=raw_line.get_color(), degree=trend_degree)

    ax.set_title(
        f"Validation Spearman Correlation (ρ) per Epoch{title_suffix}",
        fontsize=13,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Spearman Correlation (ρ)", fontsize=11)
    ax.legend(title="Configuration", frameon=True, fontsize=8.5, title_fontsize=10, loc="best")
    fig.tight_layout()

    spearman_filename = f"val_spearman_per_epoch{'_' + filter_split if filter_split else ''}.png"
    spearman_path = os.path.join(output_dir, spearman_filename)
    fig.savefig(spearman_path)
    plt.close(fig)

    print(f"Saved plots to:\n  - {mse_path}\n  - {spearman_path}")


if __name__ == "__main__":
    # Set trend_degree=1 for a straight line, or 2/3 for a curved learning trendline
    plot_training_curves(trend_degree=2)

    df_all = pd.read_csv(HISTORY_CSV)
    if df_all["split_type"].nunique() > 1:
        for split in df_all["split_type"].unique():
            plot_training_curves(filter_split=split, trend_degree=2)