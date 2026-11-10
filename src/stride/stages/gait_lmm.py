# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "statsmodels",
#     "pyyaml",
# ]
# ///
"""
Config-driven gait analysis pipeline (Stages 6-9).

Loads gait_per_stride_filtered.csv for each batch defined in config.yaml,
standardizes columns, tags trial phases, computes derived metrics,
runs M1/M2 LMMs, applies FDR correction, and outputs results + heatmaps.

Inputs per batch: an explicit `stride_csv` / `decisions_csv` in the batch config wins; without
them the stage reads what stages 3-6 wrote under <output>/<batch>/ (see stride.stages.batch_paths).

Usage:
    python -m stride.stages.gait_lmm --config config.yaml --output output
    python -m stride.stages.gait_lmm --config config.yaml --output output --batch my_batch
    python -m stride.stages.gait_lmm --config config.yaml --trial-phase task  # only task strides (default)
    python -m stride.stages.gait_lmm --config config.yaml --trial-phase all   # all strides
"""

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import yaml
from matplotlib.colors import Normalize
from statsmodels.stats.multitest import multipletests

from stride.stages.batch_paths import (
    StageError, batch_name, default_paths, exists, load_config_or_exit, missing_message,
    resolve_input, select_batches,
)

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIG LOADING
# ============================================================================

def load_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================================
# DATA LOADING & STANDARDIZATION
# ============================================================================

def genotype_levels(config: dict) -> list:
    """Genotype levels in model order: the reference first, then the others as listed.

    With the shipped settings (reference WT, order WT / APP / Tau) this is WT, APP, Tau - the
    levels this stage has always used.
    """
    stats_cfg = config.get("statistics", {}) or {}
    order = list(stats_cfg.get("genotype_order") or ["WT", "APP", "Tau"])
    ref = stats_cfg.get("reference_genotype", "WT")
    return [ref] + [g for g in order if g != ref]


def load_batch(batch_key: str, batch_cfg: dict, config: dict | None = None,
               output_dir: Path | None = None) -> pd.DataFrame:
    """Load and standardize a single batch's stride data.

    The stride file is the batch's `stride_csv` when that path exists, otherwise the filtered
    stride file stage 6 wrote under <output>/<batch>/. Raises StageError when neither exists.
    """
    default = default_paths(output_dir, batch_key)["stride_csv"] if output_dir is not None else None
    found, source = resolve_input(batch_cfg, "stride_csv", default)
    if found is None:
        raise StageError(missing_message("The filtered stride file", batch_cfg, "stride_csv", default,
                                         "stage 4-6 (run_gait_extraction)"))
    path = str(found)
    if source == "previous stage":
        print(f"  Using pipeline output: {path}")
    print(f"  Loading {batch_key}: {path}")
    df = pd.read_csv(path, low_memory=False)

    # Handle canonical column names from gait_extraction.py (mouse_id, sex, genotype)
    if "mouse_id" in df.columns and "MouseID" not in df.columns:
        df["MouseID"] = df["mouse_id"].astype(str)
    if "genotype" in df.columns and "Genotype" not in df.columns:
        df["Genotype"] = df["genotype"]
    if "sex" in df.columns and "Sex" not in df.columns:
        df["Sex"] = df["sex"]

    # Standardize MouseID (from batch-specific columns if not already set)
    if "MouseID" not in df.columns:
        id_cols = batch_cfg.get("mouse_id_cols") or (batch_cfg.get("column_map") or {}).get("mouse_id")
        if isinstance(id_cols, str):
            id_cols = [id_cols]
        if not id_cols or any(c not in df.columns for c in id_cols):
            raise StageError(f"{path} has no mouse id column (expected `mouse_id`, `MouseID`, or the "
                             f"columns named by `mouse_id_cols` in the batch: {id_cols})")
        sep = batch_cfg.get("mouse_id_sep", "_")
        if len(id_cols) == 1:
            df["MouseID"] = df[id_cols[0]].astype(str)
        else:
            df["MouseID"] = df[id_cols[0]].astype(str)
            for col in id_cols[1:]:
                df["MouseID"] = df["MouseID"] + sep + df[col].astype(str)

    if "Genotype" not in df.columns:
        raise StageError(f"{path} has no genotype column (`genotype` / `Genotype`). The mixed models "
                         f"compare genotypes: give the batch a `metadata_csv` with Mouse and Genotype "
                         f"columns and re-run stages 3 and 4-6.")

    # Standardize Genotype
    n_all = len(df)
    df = df.dropna(subset=["Genotype"]).copy()
    if len(df) < n_all:
        print(f"    NOTE: {n_all - len(df):,} strides have no genotype and are left out")
    genotype_order = genotype_levels(config or {})
    labels = df["Genotype"].astype(str)
    unknown = labels[~labels.isin(genotype_order)]
    if len(unknown):
        print(f"    WARNING: {len(unknown):,} strides carry genotype label(s) "
              f"{sorted(unknown.unique())[:6]} that are not in statistics.genotype_order "
              f"{genotype_order}; they are left out of the models (add the label to genotype_order "
              f"to include it).")
    df["Genotype"] = pd.Categorical(df["Genotype"], categories=genotype_order)
    # Model only the genotypes that are present: an empty level would add an all-zero column
    # and make every fit singular (e.g. a WT + Tau batch with the WT / APP / Tau levels).
    df["Genotype"] = df["Genotype"].cat.remove_unused_categories()

    # Standardize Sex if available
    if batch_cfg.get("has_sex", False) and "Sex" in df.columns:
        df["Sex"] = pd.Categorical(df["Sex"], categories=["F", "M"])

    # Add batch label
    df["batch"] = batch_name(batch_key, batch_cfg)
    df["batch_key"] = batch_key

    n_strides = len(df)
    n_mice = df["MouseID"].nunique()
    print(f"    {n_strides:,} strides, {n_mice} mice")

    return df


def compute_derived_metrics(df: pd.DataFrame, metrics_cfg: list) -> pd.DataFrame:
    """Compute any derived metrics (e.g., step_freq_hz)."""
    for m in metrics_cfg:
        col = m["col"]
        if "derived" in m and col not in df.columns:
            expr = m["derived"]
            # Simple evaluation: "1000 / duration_ms"
            if "/" in expr:
                parts = expr.split("/")
                numerator = float(parts[0].strip())
                denominator_col = parts[1].strip()
                if denominator_col in df.columns:
                    df[col] = numerator / df[denominator_col]
                    print(f"    Computed {col} = {expr}")
        elif "derived" in m and col in df.columns:
            pass  # already exists
    return df


# ============================================================================
# TRIAL PHASE TAGGING (Stage 7)
# ============================================================================

def tag_trial_phases(df: pd.DataFrame, batch_cfg: dict) -> pd.DataFrame:
    """Tag each stride with trial_phase using events from T-maze behavioral analysis.

    Tags:
      - 'task': stride occurs between gate_frame and entry_frame
      - 'return': stride occurs outside valid trial window
      - 'poked_return': return stride with abnormally high speed
      - 'unmatched': stride could not be matched to any trial
    """
    decisions_path = batch_cfg.get("decisions_csv")
    if not decisions_path or not Path(decisions_path).exists():
        # No explicit (existing) path in the batch: use what stage 3 wrote for this batch.
        fallback = batch_cfg.get("_decisions_csv_default")
        if fallback and exists(fallback):
            decisions_path = str(fallback)
            print(f"    Using pipeline output: {decisions_path}")
    if not decisions_path or not Path(decisions_path).exists():
        print(f"    WARNING: no decisions.csv found — strides cannot be tagged as task / return "
              f"(set `decisions_csv` in the batch or run stage 3); all strides are 'untagged'")
        df["trial_phase"] = "untagged"
        return df

    decisions = pd.read_csv(decisions_path, low_memory=False)
    print(f"    Loaded {len(decisions)} trials from {Path(decisions_path).name}")

    # Normalize video paths for matching (use filename stem)
    def normalize_path(p):
        if pd.isna(p):
            return ""
        return Path(str(p)).stem

    decisions["_video_key"] = decisions["video_path"].apply(normalize_path)
    df["_video_key"] = df["video_path"].apply(normalize_path) if "video_path" in df.columns else ""

    # Initialize
    df["trial_phase"] = "unmatched"

    # For each stride, check if it falls within a valid trial window
    for video_key in df["_video_key"].unique():
        if not video_key:
            continue
        stride_mask = df["_video_key"] == video_key
        trial_rows = decisions[decisions["_video_key"] == video_key]

        if trial_rows.empty:
            continue

        for _, trial in trial_rows.iterrows():
            gate = trial.get("gate_frame")
            entry = trial.get("entry_frame")
            if pd.isna(gate) or pd.isna(entry):
                continue

            # Strides within valid trial window
            task_mask = stride_mask & (df["frame_start"] >= gate) & (df["frame_start"] <= entry)
            df.loc[task_mask, "trial_phase"] = "task"

        # Everything not tagged as 'task' in this video is 'return'
        return_mask = stride_mask & (df["trial_phase"] == "unmatched")
        df.loc[return_mask, "trial_phase"] = "return"

    # Detect poked_return: return strides with speed > 2x mouse's task median
    for mouse_id in df["MouseID"].unique():
        mouse_mask = df["MouseID"] == mouse_id
        task_strides = df[mouse_mask & (df["trial_phase"] == "task")]
        return_strides = df[mouse_mask & (df["trial_phase"] == "return")]

        if task_strides.empty or return_strides.empty:
            continue
        if "stride_speed_cm_s_mean" not in df.columns:
            continue

        task_median_speed = task_strides["stride_speed_cm_s_mean"].median()
        if pd.isna(task_median_speed) or task_median_speed <= 0:
            continue

        speed_threshold = task_median_speed * 2.0
        poked_mask = (
            mouse_mask
            & (df["trial_phase"] == "return")
            & (df["stride_speed_cm_s_mean"] > speed_threshold)
        )
        df.loc[poked_mask, "trial_phase"] = "poked_return"

    # Cleanup temp column
    df.drop(columns=["_video_key"], inplace=True)

    # Report
    counts = df["trial_phase"].value_counts()
    for phase, count in counts.items():
        print(f"    {phase}: {count:,} strides ({100*count/len(df):.1f}%)")

    return df


# ============================================================================
# LMM ANALYSIS (Stage 9)
# ============================================================================

def run_lmm(data: pd.DataFrame, variable: str, model_name: str,
            model_cfg: dict, batch_cfg: dict, config: dict,
            fit_log: list | None = None) -> dict | None:
    """Run a single LMM for one variable on one batch.

    Returns the result row, or None when the fit is not produced. `fit_log` (optional list)
    receives one entry per call saying whether the fit was kept and, if not, why - the fits
    that are kept and their numbers are exactly as before.
    """
    def skip(reason: str):
        if fit_log is not None:
            fit_log.append({"variable": variable, "model": model_name, "status": "skipped", "reason": reason})
        return None

    if variable not in data.columns:
        return skip("metric column not in the stride file")

    # Skip if outcome is a covariate in M2
    speed_col = config["covariates"]["speed"]
    if model_cfg.get("skip_outcome_if_covariate") and variable == speed_col:
        return skip("outcome is a covariate of this model")

    # Required columns
    covariates = model_cfg["covariates"]
    required = [variable, "Genotype", "MouseID"] + covariates
    if batch_cfg.get("has_sex", False) and "Sex" in data.columns:
        required.append("Sex")
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        return skip(f"column(s) missing in the stride file: {missing_cols}")

    subset = data[required].dropna().copy()
    if len(subset) < 100:
        return skip(f"fewer than 100 strides with all values present ({len(subset)})")

    ref_geno = config["statistics"]["reference_genotype"]
    # Only the genotypes present among these strides enter the model (no empty level).
    if hasattr(subset["Genotype"], "cat"):
        subset["Genotype"] = subset["Genotype"].cat.remove_unused_categories()
        present = [str(c) for c in subset["Genotype"].cat.categories]
        if ref_geno not in present:
            return skip(f"no strides of the reference genotype '{ref_geno}' (present: {present})")
        if len(present) < 2:
            return skip(f"only one genotype present ({present}): nothing to compare")

    # Z-score outcome and covariates
    y_mean = subset[variable].mean()
    y_std = subset[variable].std()
    if y_std == 0:
        return skip("outcome has zero variance")
    subset["y_z"] = (subset[variable] - y_mean) / y_std

    cov_terms = []
    for cov in covariates:
        z_col = f"{cov}_z"
        cov_mean = subset[cov].mean()
        cov_std = subset[cov].std()
        if cov_std == 0:
            continue
        subset[z_col] = (subset[cov] - cov_mean) / cov_std
        cov_terms.append(z_col)

    # Build formula
    formula_parts = ["y_z ~ Genotype"]
    if batch_cfg.get("has_sex", False) and "Sex" in subset.columns:
        formula_parts.append("Sex")
    formula_parts.extend(cov_terms)
    formula = " + ".join(formula_parts)

    optimizer = config["statistics"]["optimizer"]
    genotypes = [g for g in config["statistics"]["genotype_order"] if g != ref_geno]

    try:
        model = smf.mixedlm(formula, subset, groups=subset["MouseID"])
        result = model.fit(reml=True, method=optimizer)
        if result.bse.max() > 10:
            return skip(f"fit dropped: a standard error is above 10 (max {float(result.bse.max()):.3g})")

        output = {
            "variable": variable,
            "model": model_name,
            "n_obs": len(subset),
            "n_mice": subset["MouseID"].nunique(),
            "converged": True,
        }

        for geno in genotypes:
            param = f"Genotype[T.{geno}]"
            if param in result.params.index:
                output[f"{geno}_coef"] = result.params[param] * y_std
                output[f"{geno}_se"] = result.bse[param] * y_std
                output[f"{geno}_pval"] = result.pvalues[param]
                output[f"{geno}_coef_z"] = result.params[param]  # standardized

        if fit_log is not None:
            fit_log.append({"variable": variable, "model": model_name, "status": "ok", "reason": "",
                            "optimizer_converged": bool(getattr(result, "converged", True))})
        return output
    except Exception as exc:
        return skip(f"fit raised {type(exc).__name__}: {str(exc)[:160]}")


def run_all_lmms(data: pd.DataFrame, batch_key: str, batch_cfg: dict,
                 config: dict, fit_log: list | None = None) -> pd.DataFrame:
    """Run M1 and M2 for all metrics on one batch."""
    metrics = [m["col"] for m in config["metrics"]]
    pretty = {m["col"]: m["label"] for m in config["metrics"]}
    results = []
    local_log: list = []

    for model_name, model_cfg in config["models"].items():
        print(f"\n  Running {model_name} on {batch_name(batch_key, batch_cfg)}...")
        for variable in metrics:
            row = run_lmm(data, variable, model_name, model_cfg, batch_cfg, config, fit_log=local_log)
            if row:
                row["batch"] = batch_name(batch_key, batch_cfg)
                row["batch_key"] = batch_key
                row["label"] = pretty.get(variable, variable)
                results.append(row)

    # Report what was fitted and what was not (and why), instead of dropping fits silently
    n_ok = sum(1 for e in local_log if e["status"] == "ok")
    skipped = [e for e in local_log if e["status"] != "ok"]
    print(f"\n  Model fits for {batch_key}: {n_ok} kept, {len(skipped)} not produced")
    reasons: dict = {}
    for e in skipped:
        key = e["reason"].split(" (")[0] if e["reason"].startswith(("fewer than", "fit dropped")) else e["reason"]
        reasons.setdefault(key, []).append(f"{e['model']}:{e['variable']}")
    for reason, items in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
        print(f"    {len(items)} x {reason}  [{', '.join(items[:4])}{' ...' if len(items) > 4 else ''}]")
    if fit_log is not None:
        for e in local_log:
            fit_log.append({"batch_key": batch_key, **e})

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def add_fdr_columns(results: pd.DataFrame, genotypes: list) -> pd.DataFrame:
    """Add `<genotype>_fdr` columns: Benjamini-Hochberg over all genotype p-values of one
    batch x model, the same family the heatmap uses for its black / grey stars. New columns
    only (appended at the end); no existing column is touched."""
    out = results.copy()
    pcols = [f"{g}_pval" for g in genotypes if f"{g}_pval" in out.columns]
    for c in pcols:
        out[c.replace("_pval", "_fdr")] = np.nan
    if not pcols:
        return out
    for _, idx in out.groupby(["batch_key", "model"], sort=False).groups.items():
        block = out.loc[idx, pcols].to_numpy(dtype=float)
        flat = block.flatten()
        ok = ~np.isnan(flat)
        if ok.any():
            _, fdr, _, _ = multipletests(flat[ok], method="fdr_bh")
            adj = np.full_like(flat, np.nan)
            adj[ok] = fdr
            out.loc[idx, [c.replace("_pval", "_fdr") for c in pcols]] = adj.reshape(block.shape)
    return out


# ============================================================================
# HEATMAP VISUALIZATION
# ============================================================================

ROW_ORDER = [
    "Tailtip Phase",
    "Tailbase Phase",
    "Snout Phase",
    "Tailtip Lateral Disp.",
    "Tailbase Lateral Disp.",
    "Snout Lateral Disp.",
    "Stride Length",
    "Step Width",
    "Step Length",
    "Stride Duration",
    "Step Frequency",
    "Temporal Symmetry",
    "Limb Duty Factor",
    "Angular Velocity",
]


def create_heatmap(results_df: pd.DataFrame, title: str, out_path: Path,
                   config: dict):
    """Create dual-panel heatmap: -log10(p) + effect size bubbles."""
    df = results_df[results_df["converged"] == True].copy()
    if len(df) == 0:
        return

    ref_geno = config["statistics"]["reference_genotype"]
    genotypes = [g for g in config["statistics"]["genotype_order"] if g != ref_geno]

    # Build matrices
    labels = df["label"].unique()
    Hc_pval = pd.DataFrame(index=labels, columns=genotypes, dtype=float)
    Bc = pd.DataFrame(index=labels, columns=genotypes, dtype=float)

    # Deduplicate: keep first occurrence of each label
    df = df.drop_duplicates(subset=["label"], keep="first")
    labels = df["label"].unique()
    Hc_pval = pd.DataFrame(index=labels, columns=genotypes, dtype=float)
    Bc = pd.DataFrame(index=labels, columns=genotypes, dtype=float)

    for _, row in df.iterrows():
        label = row["label"]
        for geno in genotypes:
            Hc_pval.at[label, geno] = row.get(f"{geno}_pval", np.nan)
            Bc.at[label, geno] = row.get(f"{geno}_coef_z", np.nan)

    Hc = -np.log10(Hc_pval.astype(float).clip(lower=np.finfo(float).tiny))

    # FDR correction
    all_pvals = Hc_pval.values.flatten()
    valid_mask = ~np.isnan(all_pvals)
    if valid_mask.any():
        _, fdr_vals, _, _ = multipletests(all_pvals[valid_mask], method="fdr_bh")
        all_fdr = np.full_like(all_pvals, np.nan)
        all_fdr[valid_mask] = fdr_vals
        Hc_FDR = pd.DataFrame(all_fdr.reshape(Hc_pval.shape),
                               index=Hc_pval.index, columns=Hc_pval.columns)
    else:
        Hc_FDR = Hc_pval.copy()

    # Reorder rows
    row_order = [r for r in ROW_ORDER if r in Hc.index]
    remaining = [r for r in Hc.index if r not in row_order]
    row_order = row_order + remaining
    Hc = Hc.reindex(row_order)
    Bc = Bc.reindex(row_order)
    Hc_pval = Hc_pval.reindex(row_order)
    Hc_FDR = Hc_FDR.reindex(row_order)

    nrows, ncols = len(Hc), len(Hc.columns)

    # --- Draw ---
    fig = plt.figure(figsize=(4.2, max(5, nrows * 0.38 + 1.2)), dpi=200)
    left_margin, panel_width, gap = 0.28, 0.26, 0.01
    bottom, height = 0.12, 0.75

    axL = fig.add_axes([left_margin, bottom, panel_width, height])
    axR = fig.add_axes([left_margin + panel_width + gap, bottom, panel_width, height])

    # LEFT: -log10(p)
    YELLOW = "#fff3b0"
    sns.heatmap(np.ones((nrows, ncols))[::-1], ax=axL, cmap=[YELLOW],
                vmin=0, vmax=1, cbar=False, linewidths=1.0, linecolor="white")

    pos = Hc.fillna(0).astype(float)
    vmax_pos = max(5.0, float(np.nanmax(pos.values)))
    sns.heatmap(pos.iloc[::-1].clip(lower=1.0), ax=axL,
                mask=(pos <= 0).iloc[::-1].values,
                cmap="YlGnBu", vmin=1.0, vmax=vmax_pos,
                linewidths=1.0, linecolor="white", cbar=False)

    for i, pheno in enumerate(Hc.index[::-1]):
        for j, col in enumerate(Hc.columns):
            p_val = float(Hc_pval.at[pheno, col])
            fdr_val = float(Hc_FDR.at[pheno, col])
            if not np.isfinite(p_val):
                continue
            stars = "***" if p_val < 1e-3 else ("**" if p_val < 1e-2 else ("*" if p_val < 5e-2 else ""))
            if stars:
                color = "black" if fdr_val < 0.05 else "#555555"
                axL.text(j + 0.5, i + 0.5, stars, ha="center", va="center",
                         color=color, fontsize=11, fontweight="bold")

    for sp in axL.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color("black")
    axL.set_xticklabels(Hc.columns, rotation=0, fontsize=10)
    axL.set_yticklabels(Hc.index[::-1], rotation=0, fontsize=9)
    axL.set_title(u"\u2212log\u2081\u2080(p)", fontsize=10, fontweight="bold", pad=8)

    # RIGHT: Effect sizes
    sns.heatmap(np.zeros((nrows, ncols)), ax=axR, cmap="Greys",
                vmin=0, vmax=1, cbar=False, linewidths=1.0, linecolor="white")

    vals = Bc.values.astype(float)
    finite_vals = vals[np.isfinite(vals)]
    beta_lim = np.nanpercentile(np.abs(finite_vals), 95) if len(finite_vals) > 0 else 1.0
    beta_lim = max(beta_lim, 0.1)
    norm = Normalize(vmin=-beta_lim, vmax=beta_lim)
    cmap_effect = plt.get_cmap("RdBu_r")

    for i in range(nrows):
        for j in range(ncols):
            b = vals[nrows - 1 - i, j]
            if np.isfinite(b):
                size = 250 * (abs(b) / beta_lim) ** 0.85 + 35
                axR.scatter(j + 0.5, i + 0.5, s=size, color=cmap_effect(norm(b)),
                            alpha=0.9, edgecolors="black", linewidths=0.5)

    for sp in axR.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.2); sp.set_color("black")
    axR.set_xticklabels(Bc.columns, rotation=0, fontsize=10)
    axR.set_yticklabels([]); axR.tick_params(axis="y", left=False)
    axR.set_title(u"Effect (\u03b2/SD)", fontsize=10, fontweight="bold", pad=8)

    # Colorbars
    cbar_left, cbar_width = 0.85, 0.015
    cax1 = fig.add_axes([cbar_left, 0.52, cbar_width, 0.35])
    sm1 = plt.cm.ScalarMappable(cmap="YlGnBu", norm=Normalize(vmin=1.0, vmax=vmax_pos))
    sm1.set_array([])
    plt.colorbar(sm1, cax=cax1).set_label(r"$-\log_{10}(p)$", fontsize=8)

    cax2 = fig.add_axes([cbar_left, 0.12, cbar_width, 0.35])
    sm2 = plt.cm.ScalarMappable(norm=norm, cmap=cmap_effect)
    sm2.set_array([])
    plt.colorbar(sm2, cax=cax2).set_label(u"\u03b2 (SD)", fontsize=8)

    n_obs = df["n_obs"].iloc[0]
    n_mice = df["n_mice"].iloc[0]
    fig.suptitle(f"{title}\nn = {n_obs:,} strides, {n_mice} mice",
                 fontsize=9, fontweight="bold", y=0.98)

    for ext in ["png", "pdf"]:
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {out_path.name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Config-driven gait LMM analysis")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: ./config.yaml)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: ./output)")
    parser.add_argument("--batch", default=None, help="Process only this batch key")
    parser.add_argument("--trial-phase", default="task",
                        choices=["task", "all", "return", "poked_return"],
                        help="Which trial_phase to include (default: task)")
    args = parser.parse_args()

    config = load_config_or_exit(args.config)
    for section in ("metrics", "models", "statistics", "covariates"):
        if section not in config:
            print(f"ERROR: {args.config} has no `{section}:` section (needed by the mixed-model stage; "
                  f"copy it from config.example.yaml).")
            raise SystemExit(2)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Gait Analysis Pipeline — Stages 7-9")
    print(f"Config: {args.config}")
    print(f"Output: {out_dir}")
    print(f"Trial phase filter: {args.trial_phase}")
    print("=" * 60)

    all_results = []
    fit_log: list = []
    problems: list = []          # why a batch gave no result
    inputs_used: dict = {}

    for batch_key, batch_cfg in select_batches(config, args.batch):
        print(f"\n{'='*60}")
        print(f"Batch: {batch_name(batch_key, batch_cfg)}")
        print(f"{'='*60}")

        # Stage 6: Load filtered strides
        try:
            df = load_batch(batch_key, batch_cfg, config, out_dir)
        except StageError as exc:
            print(f"  ERROR: {exc}")
            problems.append(f"{batch_key}: {exc}")
            continue

        # Stage 7: Trial phase tagging
        print("\n  Stage 7: Trial phase tagging...")
        tag_cfg = dict(batch_cfg)
        tag_cfg["_decisions_csv_default"] = default_paths(out_dir, batch_key)["decisions_csv"]
        df = tag_trial_phases(df, tag_cfg)

        # Filter by trial phase
        if args.trial_phase != "all":
            before = len(df)
            df = df[df["trial_phase"] == args.trial_phase].copy()
            print(f"  Filtered to trial_phase='{args.trial_phase}': {before:,} -> {len(df):,} strides")
            if before and len(df) == 0:
                problems.append(f"{batch_key}: no '{args.trial_phase}' strides (of {before:,}); strides are tagged "
                                f"from decisions.csv (gate_frame..entry_frame) - check `decisions_csv`, or use "
                                f"--trial-phase all")

        # Stage 8: Derived metrics
        print("\n  Stage 8: Derived metrics...")
        df = compute_derived_metrics(df, config["metrics"])

        # Stage 9: LMM
        print("\n  Stage 9: LMM analysis...")
        n_log = len(fit_log)
        results = run_all_lmms(df, batch_key, batch_cfg, config, fit_log=fit_log)
        inputs_used[batch_key] = {"strides": int(len(df)), "mice": int(df["MouseID"].nunique()) if len(df) else 0,
                                  "genotypes": sorted(map(str, df["Genotype"].dropna().unique())) if len(df) else []}
        if len(results) > 0:
            all_results.append(results)

            # Generate heatmaps per model
            for model_name in config["models"]:
                model_results = results[results["model"] == model_name]
                if len(model_results) > 0:
                    title = f"{model_name} — {batch_name(batch_key, batch_cfg)}"
                    heatmap_path = out_dir / f"heatmap_{model_name}_{batch_key}"
                    create_heatmap(model_results, title, heatmap_path, config)
        else:
            why = sorted({e["reason"] for e in fit_log[n_log:] if e["status"] != "ok"})
            problems.append(f"{batch_key}: no model could be fitted ({'; '.join(why[:3]) or 'no strides'})")

    # What was fitted and what was not, per batch / model / metric
    if fit_log:
        pd.DataFrame(fit_log).to_csv(out_dir / "lmm_fit_log.csv", index=False)
        print(f"\nFit log (every metric x model, kept or not and why): {out_dir / 'lmm_fit_log.csv'}")

    ref_geno = config["statistics"]["reference_genotype"]
    genotypes = [g for g in config["statistics"]["genotype_order"] if g != ref_geno]
    csv_path = out_dir / "lmm_results_all.csv"

    # Save combined results
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = add_fdr_columns(combined, genotypes)
        combined.to_csv(csv_path, index=False)
        print(f"\nSaved combined results: {csv_path}")
        print(f"Total: {len(combined)} model fits across {combined['batch'].nunique()} batches")

        # Print significance summary
        print("\n" + "=" * 60)
        print("SIGNIFICANCE SUMMARY (p < 0.05)")
        print("=" * 60)
        for _, row in combined.iterrows():
            for geno in genotypes:
                p = row.get(f"{geno}_pval", 1.0)
                if pd.notna(p) and p < 0.05:
                    stars = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
                    coef = row.get(f"{geno}_coef", 0)
                    print(f"  {row['batch']} | {row['model']} | {row['label']:25s} | "
                          f"{geno} vs {ref_geno}: coef={coef:+.4f}, p={p:.2e} {stars}")
    else:
        print("\nNo results generated.")

    # Provenance sidecar (best-effort; never breaks a completed run)
    try:
        from stride.run_manifest import write_run_manifest
        write_run_manifest(
            out_dir, "lmm",
            params={"trial_phase": args.trial_phase, "batch": args.batch,
                    "statistics": config.get("statistics"), "models": config.get("models"),
                    "metrics": [m["col"] for m in config["metrics"]]},
            inputs=inputs_used or None,
            outputs=[csv_path, out_dir / "lmm_fit_log.csv"],
            results={"fits_kept": sum(1 for e in fit_log if e["status"] == "ok"),
                     "fits_not_produced": sum(1 for e in fit_log if e["status"] != "ok"),
                     "problems": problems or None},
        )
    except Exception:
        pass

    if problems:
        print(f"\nStage 7-9 {'FAILED' if not all_results else 'finished with problems'}:")
        for pr in problems:
            print(f"  - {pr}")
        raise SystemExit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
