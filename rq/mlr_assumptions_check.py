#!/usr/bin/env python3
"""
MLR assumption checks for BLUE/CLM diagnostics.

This script runs basic diagnostics on an OLS regression with optional
entity and time fixed effects using within-transformed data. It reports
key statistics and simple tests that help assess Gauss-Markov/CLM conditions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _within_transform(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
    time_col: str,
    time_fe: bool,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str], np.ndarray]:
    y = panel[y_col].astype(float)
    x = panel[x_cols].astype(float)

    entity_mean_y = y.groupby(panel[entity_col]).transform("mean")
    y_tilde = y - entity_mean_y

    x_tilde = np.column_stack(
        [
            (x[c] - x[c].groupby(panel[entity_col]).transform("mean")).to_numpy()
            for c in x_cols
        ]
    )

    if time_fe:
        time_mean_y = y.groupby(panel[time_col]).transform("mean")
        overall_y = float(y.mean())
        y_tilde = y_tilde - (time_mean_y - overall_y)

        for i, c in enumerate(x_cols):
            time_mean_x = x[c].groupby(panel[time_col]).transform("mean")
            overall_x = float(x[c].mean())
            x_tilde[:, i] = x_tilde[:, i] - (time_mean_x - overall_x).to_numpy()

    col_std = np.std(x_tilde, axis=0)
    keep_cols = col_std > 1e-12
    kept_names = [name for name, keep in zip(x_cols, keep_cols) if keep]
    dropped_names = [name for name, keep in zip(x_cols, keep_cols) if not keep]

    if not np.all(keep_cols):
        x_tilde = x_tilde[:, keep_cols]

    return y_tilde.to_numpy(), x_tilde, kept_names, dropped_names, keep_cols


def _ols_fit(y: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    xtx = x.T @ x
    xtx_inv = np.linalg.pinv(xtx)
    beta = xtx_inv @ (x.T @ y)
    y_hat = x @ beta
    resid = y - y_hat

    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - (sse / sst if sst > 0 else np.nan)

    return {
        "beta": beta,
        "resid": resid,
        "y_hat": y_hat,
        "r2": r2,
        "sse": sse,
    }


def _vif(x: np.ndarray, names: list[str]) -> dict[str, float]:
    vifs: dict[str, float] = {}
    n, k = x.shape
    if k <= 1:
        for name in names:
            vifs[name] = float("nan")
        return vifs

    for i, name in enumerate(names):
        mask = [j for j in range(k) if j != i]
        xi = x[:, i]
        x_others = x[:, mask]

        fit = _ols_fit(xi, x_others)
        r2 = fit["r2"]
        if r2 is None or np.isnan(r2) or r2 >= 1.0:
            vifs[name] = float("inf")
        else:
            vifs[name] = float(1.0 / (1.0 - r2))

    return vifs


def _breusch_pagan(resid: np.ndarray, x: np.ndarray) -> dict[str, Any]:
    n, k = x.shape
    z = resid**2
    x_bp = np.column_stack([np.ones(n), x])
    fit = _ols_fit(z, x_bp)
    r2 = fit["r2"]
    lm = float(n * r2) if r2 is not None else float("nan")

    p_value = None
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.chi2.sf(lm, k))
    except Exception:
        p_value = None

    return {"lm": lm, "df": int(k), "p_value": p_value}


def _jarque_bera(resid: np.ndarray) -> dict[str, Any]:
    n = len(resid)
    if n == 0:
        return {"jb": float("nan"), "p_value": None}

    mean = float(np.mean(resid))
    std = float(np.std(resid, ddof=0))
    if std == 0:
        return {"jb": float("nan"), "p_value": None}

    skew = float(np.mean(((resid - mean) / std) ** 3))
    kurt = float(np.mean(((resid - mean) / std) ** 4))
    jb = float(n / 6.0 * (skew**2 + (kurt - 3.0) ** 2 / 4.0))

    p_value = None
    try:
        from scipy import stats  # type: ignore

        p_value = float(stats.chi2.sf(jb, 2))
    except Exception:
        p_value = None

    return {"jb": jb, "p_value": p_value, "skew": skew, "kurtosis": kurt}


def _durbin_watson(resid: np.ndarray) -> float:
    if len(resid) < 2:
        return float("nan")
    diff = np.diff(resid)
    return float(np.sum(diff**2) / np.sum(resid**2)) if np.sum(resid**2) > 0 else float("nan")


def _max_pairwise_corr(x: np.ndarray) -> float:
    if x.shape[1] <= 1:
        return float("nan")
    corr = np.corrcoef(x, rowvar=False)
    np.fill_diagonal(corr, 0.0)
    return float(np.nanmax(np.abs(corr)))


def run_checks(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    entity_col: str,
    time_col: str,
    time_fe: bool,
) -> dict[str, Any]:
    y_tilde, x_tilde, kept_cols, dropped_cols, keep_cols = _within_transform(
        panel=panel,
        y_col=y_col,
        x_cols=x_cols,
        entity_col=entity_col,
        time_col=time_col,
        time_fe=time_fe,
    )

    keep_mask = np.isfinite(y_tilde) & np.all(np.isfinite(x_tilde), axis=1)
    y_tilde = y_tilde[keep_mask]
    x_tilde = x_tilde[keep_mask]

    if x_tilde.shape[1] == 0:
        raise RuntimeError("All regressors dropped after within transform.")

    fit = _ols_fit(y_tilde, x_tilde)
    resid = fit["resid"]

    vif = _vif(x_tilde, kept_cols)
    bp = _breusch_pagan(resid, x_tilde)
    jb = _jarque_bera(resid)

    panel_keep = panel.loc[pd.Series(keep_mask, index=panel.index)]
    panel_keep = panel_keep.sort_values([time_col, entity_col])
    resid_sorted = pd.Series(resid, index=panel_keep.index).loc[panel_keep.index].to_numpy()
    dw = _durbin_watson(resid_sorted)

    condition_number = float(np.linalg.cond(x_tilde))
    max_corr = _max_pairwise_corr(x_tilde)

    return {
        "n_obs": int(len(y_tilde)),
        "n_regressors": int(x_tilde.shape[1]),
        "time_fixed_effects": bool(time_fe),
        "dropped_regressors": dropped_cols,
        "residual_mean": float(np.mean(resid)),
        "residual_std": float(np.std(resid, ddof=1)),
        "r2_within": float(fit["r2"]),
        "condition_number": condition_number,
        "max_pairwise_corr": max_corr,
        "vif": vif,
        "breusch_pagan": bp,
        "jarque_bera": jb,
        "durbin_watson": dw,
        "notes": [
            "Durbin-Watson is approximate for panel data; interpret cautiously.",
            "Breusch-Pagan and Jarque-Bera p-values require scipy if available.",
        ],
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "mlr_assumptions.json", "w", encoding="ascii") as f:
        json.dump(payload, f, indent=2)

    md_lines = [
        "# MLR Assumption Checks",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Observations | {payload.get('n_obs')} |",
        f"| Regressors | {payload.get('n_regressors')} |",
        f"| Time fixed effects | {payload.get('time_fixed_effects')} |",
        f"| Dropped regressors | {', '.join(payload.get('dropped_regressors', [])) or 'None'} |",
        f"| Residual mean | {payload.get('residual_mean')} |",
        f"| Residual std | {payload.get('residual_std')} |",
        f"| Within R2 | {payload.get('r2_within')} |",
        f"| Condition number | {payload.get('condition_number')} |",
        f"| Max pairwise corr | {payload.get('max_pairwise_corr')} |",
        f"| Durbin-Watson | {payload.get('durbin_watson')} |",
        "",
        "## Breusch-Pagan",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| LM | {payload.get('breusch_pagan', {}).get('lm')} |",
        f"| df | {payload.get('breusch_pagan', {}).get('df')} |",
        f"| p value | {payload.get('breusch_pagan', {}).get('p_value')} |",
        "",
        "## Jarque-Bera",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| JB | {payload.get('jarque_bera', {}).get('jb')} |",
        f"| p value | {payload.get('jarque_bera', {}).get('p_value')} |",
        f"| Skew | {payload.get('jarque_bera', {}).get('skew')} |",
        f"| Kurtosis | {payload.get('jarque_bera', {}).get('kurtosis')} |",
        "",
        "## VIF",
        "",
        "| Variable | VIF |",
        "|---|---:|",
    ]

    for name, value in payload.get("vif", {}).items():
        md_lines.append(f"| {name} | {value} |")

    md_lines.extend(["", "## Notes", ""])
    for note in payload.get("notes", []):
        md_lines.append(f"- {note}")

    with open(output_dir / "mlr_assumptions.md", "w", encoding="ascii") as f:
        f.write("\n".join(md_lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLR assumption checks for BLUE/CLM diagnostics.")

    parser.add_argument(
        "--panel-csv",
        type=str,
        default="results/rq1/rq1_daily_panel.csv",
        help="Panel CSV (from rq-1 pipeline).",
    )
    parser.add_argument("--y-col", type=str, default="daily_stance")
    parser.add_argument(
        "--x-cols",
        type=str,
        default="post,article_volume",
        help="Comma-separated regressor names.",
    )
    parser.add_argument("--entity-col", type=str, default="company_id")
    parser.add_argument("--time-col", type=str, default="date")
    parser.add_argument(
        "--time-fe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include time fixed effects (delta_t) in the regression.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/rq1/assumptions",
        help="Directory to save outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    panel = pd.read_csv(args.panel_csv)

    panel[args.time_col] = pd.to_datetime(panel[args.time_col], errors="coerce")
    panel = panel.dropna(subset=[args.time_col])

    x_cols = [col.strip() for col in args.x_cols.split(",") if col.strip()]

    payload = run_checks(
        panel=panel,
        y_col=args.y_col,
        x_cols=x_cols,
        entity_col=args.entity_col,
        time_col=args.time_col,
        time_fe=args.time_fe,
    )

    write_outputs(Path(args.output_dir), payload)
    print("Done.")
    print(f"Outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
