
import argparse
import re
import sys
import pandas as pd


def find_metric_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def extract_fold(run_name: str):
    if pd.isna(run_name):
        return None
    m = re.search(r'fold[_\- ]?(\d+)', str(run_name), flags=re.IGNORECASE)
    return int(m.group(1)) if m else None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "For each run, compute fsum = f10 + f25 + f50, "
            "then keep the epoch/step with the maximum fsum."
        )
    )
    parser.add_argument("input_csv", help="Path to the input CSV.")
    parser.add_argument(
        "-o", "--output-csv",
        default="best_fsum_per_run.csv",
        help="Path to the output CSV."
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)

    run_col = find_metric_column(df.columns, ["run_name", "run", "name"])
    step_col = find_metric_column(df.columns, ["epoch", "_step", "step"])

    f10_col = find_metric_column(df.columns, [
        "metrics/val/f1_10", "f1_10", "f10"
    ])
    f25_col = find_metric_column(df.columns, [
        "metrics/val/f1_25", "f1_25", "f_25", "f25"
    ])
    f50_col = find_metric_column(df.columns, [
        "metrics/val/f1_50", "f1_50", "f50"
    ])

    missing = []
    if run_col is None:
        missing.append("run_name")
    if step_col is None:
        missing.append("epoch/_step")
    if f10_col is None:
        missing.append("f1_10")
    if f25_col is None:
        missing.append("f1_25")
    if f50_col is None:
        missing.append("f1_50")

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only rows that look like actual metric rows.
    work = df.copy()
    work = work[work[run_col].notna()].copy()
    work = work[~work[run_col].astype(str).str.startswith("===", na=False)].copy()

    # Ensure numeric metrics.
    for col in [step_col, f10_col, f25_col, f50_col]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    # Rows need all three metrics to compute fsum.
    work = work.dropna(subset=[f10_col, f25_col, f50_col]).copy()

    work["fsum_computed"] = work[f10_col] + work[f25_col] + work[f50_col]
    work["fold"] = work[run_col].apply(extract_fold)

    # Pick the row with the highest fsum per run.
    idx = work.groupby(run_col)["fsum_computed"].idxmax()
    best = work.loc[idx].copy()

    # Stable sort for readability.
    sort_cols = [c for c in ["fold", run_col, step_col] if c in best.columns]
    if sort_cols:
        best = best.sort_values(sort_cols).reset_index(drop=True)

    output = pd.DataFrame({
        "fold": best["fold"],
        "run_name": best[run_col],
        "best_step": best[step_col],
        "f10": best[f10_col],
        "f25": best[f25_col],
        "f50": best[f50_col],
        "fsum": best["fsum_computed"],
    })

    output.to_csv(args.output_csv, index=False)

    print(f"Wrote {len(output)} rows to {args.output_csv}")
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
