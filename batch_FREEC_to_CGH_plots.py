#!/usr/bin/env python
"""
Batch Control-FREEC CN plots
 - Moving-average (51-bin) of log2 ratios, centered to CN=2
 - Autosomes only (1..22), X/Y removed
 - Y-axis fixed to [-10, 10]
 - Baseline at 0, alternating faint background bands
 - Raw points and segments hidden (MA only)
 - Excludes samples containing 'ARPE19_P27' (case-insensitive)
 - Recurses through the input folder and processes all *_ratio.txt + *_CNVs pairs

"""

import sys, re
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ----- Settings -----
MA_WINDOW = 51                     # moving-average window (bins)
MIN_VALID_FRAC = 0.60              # require >=60% valid points in MA window to draw
Y_MIN, Y_MAX = -7.0, 7.0         # fixed y-range
LINE_COLOR = "#d2691e"             # dark orange
OUT_SUBDIR = "freec_plots_Y7"
EXCLUDE_PATTERN = re.compile(r"ARPE19_P27", re.IGNORECASE)
AUTOSOMES = [str(i) for i in range(1, 23)]
HG38_LEN = {
    "1": 248956422, "2": 242193529, "3": 198295559, "4": 190214555, "5": 181538259,
    "6": 170805979, "7": 159345973, "8": 145138636, "9": 138394717, "10": 133797422,
    "11": 135086622, "12": 133275309, "13": 114364328, "14": 107043718, "15": 101991189,
    "16": 90338345,  "17": 83257441,  "18": 80373285,  "19": 58617616,  "20": 64444167,
    "21": 46709983,  "22": 50818468
}

# FREEC header expectations (match files exactly)
RATIO_COLS = [
    "Chromosome","Start","Ratio","MedianRatio","CopyNumber","BAF",
    "estimatedBAF","Genotype","UncertaintyOfGT","Subclone_CN","Subclone_Population"
]
CNVS_COLS  = [
    "chromosome","start position","end position","predicited copy number",
    "type of alteration","genotype","genotype percentage of uncertainty"
]

def moving_average_strict(series: pd.Series, k=MA_WINDOW, min_valid_frac=MIN_VALID_FRAC) -> pd.Series:
    """Rolling mean requiring a minimum fraction of valid (non-NaN) points."""
    if k % 2 == 0:
        k += 1
    min_valid = int(np.ceil(k * float(min_valid_frac)))
    return series.rolling(window=k, center=True, min_periods=min_valid).mean()

def read_freec_ratio(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=0, names=RATIO_COLS, usecols=RATIO_COLS,
        dtype={"Chromosome":"string","Start":"Int64","Ratio":"float64"},
        na_values=["-1","-","NA","NaN"], keep_default_na=True, engine="c"
    )
    df = df.rename(columns={"Chromosome":"chr","Start":"start","Ratio":"ratio"})
    df["chr"] = df["chr"].astype(str).str.replace("^chr","",regex=True).str.upper()
    df = df[df["chr"].isin(AUTOSOMES)].copy()  # autosomes only
    order_key = {c: i for i, c in enumerate(AUTOSOMES)}
    df["__k__"] = df["chr"].map(order_key)
    df = df.sort_values(["__k__", "start"]).drop(columns="__k__").reset_index(drop=True)
    return df

def read_freec_cnvs(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path, sep="\t", header=0, names=CNVS_COLS, usecols=CNVS_COLS,
        dtype={"chromosome":"string","start position":"Int64","end position":"Int64","predicited copy number":"float64"},
        na_values=["-1","-","NA","NaN"], keep_default_na=True, engine="c"
    )
    df = df.rename(columns={
        "chromosome":"chr","start position":"start","end position":"end","predicited copy number":"cn"
    })
    df["chr"] = df["chr"].astype(str).str.replace("^chr","",regex=True).str.upper()
    df = df[df["chr"].isin(AUTOSOMES)].copy()
    order_key = {c: i for i, c in enumerate(AUTOSOMES)}
    df["__k__"] = df["chr"].map(order_key)
    df = df.sort_values(["__k__", "start", "end"]).drop(columns="__k__").reset_index(drop=True)
    return df

def infer_step_and_mid(ratio_df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    out = ratio_df.copy()
    steps = []
    for _, sub in out.groupby("chr", sort=False):
        d = sub["start"].astype("float64").diff().dropna()
        d = d[d > 0]
        if len(d) > 0:
            steps.append(np.median(d.values))
    step = int(np.median(steps)) if steps else 50000
    if step <= 0:
        step = 50000
    out["pos"] = out["start"].astype("float64") + step/2.0
    return out, step

def baseline_ratio_cn2(ratio_df: pd.DataFrame, cnv_df: pd.DataFrame) -> float:
    """Estimate neutral ratio using bins inside CN≈2 segments; robust to zeros/invalids."""
    neutral = cnv_df[np.round(cnv_df["cn"]).astype("Int64") == 2]
    vals = []
    for c, bins in ratio_df.groupby("chr", sort=False):
        segs = neutral[neutral["chr"] == c]
        if segs.empty:
            continue
        starts = bins["start"].values
        for _, seg in segs.iterrows():
            m = (starts >= seg["start"]) & (starts <= seg["end"])
            if m.any():
                v = pd.to_numeric(bins.loc[m, "ratio"], errors="coerce")
                v = v[(v > 0) & np.isfinite(v)]
                if v.size:
                    vals.append(v.values)
    if vals:
        br = float(np.median(np.concatenate(vals)))
    else:
        v = pd.to_numeric(ratio_df["ratio"], errors="coerce")
        v = v[(v > 0) & np.isfinite(v)]
        br = float(np.median(v)) if v.size else float("nan")
    if not np.isfinite(br) or br <= 0:
        v = pd.to_numeric(ratio_df["ratio"], errors="coerce")
        v = v[(v > 0) & np.isfinite(v)]
        br = float(np.median(v)) if v.size else 1.0
    return br

def concat_with_gaps(x_list, y_list):
    xs, ys = [], []
    for x, y in zip(x_list, y_list):
        xs.append(x); ys.append(y)
        xs.append(np.array([np.nan])); ys.append(np.array([np.nan]))
    return np.concatenate(xs), np.concatenate(ys)

def plot_ma_autosomes(sample_name: str, ratio_df: pd.DataFrame, out_dir: Path):
    # layout across concatenated chromosomes
    cum_start, cum = {}, 0
    for c in AUTOSOMES:
        cum_start[c] = cum
        fallback = int(ratio_df.loc[ratio_df["chr"] == c, "start"].max() or 0)
        cum += HG38_LEN.get(c, fallback)

    # build concatenated MA line
    x_ma, y_ma = [], []
    for c, sub in ratio_df.groupby("chr", sort=False):
        x = cum_start[c] + sub["pos"].astype(float).values
        x_ma.append(x)
        y_ma.append(sub["log2_ma"].astype(float).values)
    X, Y = concat_with_gaps(x_ma, y_ma)

    # draw
    fig = plt.figure(figsize=(18, 5.6))
    ax  = fig.add_axes([0.06, 0.10, 0.92, 0.86])

    # faint alternating background + separators
    for i, c in enumerate(AUTOSOMES):
        x0 = cum_start[c]; x1 = cum_start[c] + HG38_LEN.get(c, 0)
        if i % 2 == 0:
            ax.axvspan(x0, x1, alpha=0.06)
        ax.axvline(x1, linewidth=0.5)

    # baseline
    ax.axhline(0, linewidth=1.0)

    # preserve NaNs so gaps appear
    X_plot, Y_plot = X.copy(), Y.copy()
    X_plot[~np.isfinite(X_plot)] = np.nan
    Y_plot[~np.isfinite(Y_plot)] = np.nan
    ax.plot(X_plot, Y_plot, linewidth=0.9, color=LINE_COLOR)

    # ticks/labels: keep chromosome labels, remove y-axis stuff, remove x-axis title
    xticks = [(cum_start[c] + cum_start[c] + HG38_LEN.get(c, 0))/2.0 for c in AUTOSOMES]
    ax.set_xticks(xticks)
    ax.set_xticklabels(AUTOSOMES)
    ax.set_xlim(0, cum)
    ax.set_ylim(Y_MIN, Y_MAX)

    # —— Plot items removals ——
    ax.set_xlabel("")  # remove X-axis title
    ax.set_ylabel("")  # remove Y-axis title
    ax.set_yticks([])  # remove Y-axis ticks
    ax.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

    ax.set_title(f"{sample_name} — Moving Average (autosomes only, Y=[{Y_MIN}, {Y_MAX}])")
    ax.grid(False)

    out_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_name)
    png = out_dir / f"{safe}_MA_autosomes_y_pm10.png"
    pdf = out_dir / f"{safe}_MA_autosomes_y_pm10.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

def find_pairs(root: Path):
    """Yield (sample_id, ratio_path, cnvs_path) for each *_ratio.txt with matching *_CNVs."""
    ratio_files = list(root.rglob("*_ratio.txt"))
    cnv_files   = list(root.rglob("*_CNVs"))

    def cnv_base(p: Path) -> str:
        return p.name[:-5] if p.name.endswith("_CNVs") else p.name

    cnv_map = {}
    for p in cnv_files:
        cnv_map.setdefault(cnv_base(p), []).append(p)

    for r in ratio_files:
        base = r.name.replace("_ratio.txt", "")
        candidates = cnv_map.get(base, [])
        if not candidates:
            candidates = [p for p in cnv_files if p.name.startswith(base)]
        if not candidates:
            continue
        cnv = next((p for p in candidates if p.parent == r.parent), candidates[0])
        yield base, r, cnv

def main():
    if len(sys.argv) < 2:
        print("Usage: python batch_freec_ma_plots.py \"G:\\My Drive\\...\\controlfreec\"")
        sys.exit(1)

    root = Path(sys.argv[1])
    if not root.exists():
        print(f"Input path not found: {root}")
        sys.exit(1)

    out_dir = root / OUT_SUBDIR
    n_total = n_done = 0

    for sample_id, ratio_path, cnvs_path in find_pairs(root):
        n_total += 1
        if EXCLUDE_PATTERN.search(sample_id):
            print(f"[skip] {sample_id} (excluded)")
            continue
        try:
            print(f"[run ] {sample_id}")
            r = read_freec_ratio(ratio_path)
            c = read_freec_cnvs(cnvs_path)
            if r.empty or c.empty:
                print(f"  -> empty inputs; skipping.")
                continue
            r, _ = infer_step_and_mid(r)

            # Center to CN=2 (baseline ratio) using CN=2 segments when available
            br = baseline_ratio_cn2(r, c)

            # Robust log2: drop non-positive/invalid ratios so they never appear
            ratio = pd.to_numeric(r["ratio"], errors="coerce")
            ratio = ratio.where((ratio > 0) & np.isfinite(ratio))
            r["log2ratio"] = np.log2(ratio / br)

            # Strict moving average: require >= MIN_VALID_FRAC valid points per window
            r["log2_ma"] = r.groupby("chr", sort=False)["log2ratio"].transform(
                lambda s: moving_average_strict(s, MA_WINDOW, MIN_VALID_FRAC)
            )

            # Plot MA only (with NaNs preserved to show gaps)
            plot_ma_autosomes(sample_id, r, out_dir)
            n_done += 1

        except Exception as e:
            print(f"  !! error on {sample_id}: {e}")

    print(f"Done. {n_done}/{n_total} samples processed. Outputs in: {out_dir}")

if __name__ == "__main__":
    main()


