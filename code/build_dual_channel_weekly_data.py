"""Build point-in-time weekly data for the corrected CNN + XGBoost study.

The prediction label is signal-close to the fifth following trading-day close,
minus official CSI 500 over the same closes. Execution return is next open to
the open after the label end. Stocks are never filtered using future prices.
"""
from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent.parent
RAW = Path(os.environ["MULTIFACTOR_RAW"])
OLD_PANELS = PROJECT / "work/open_factor_v1/output/panels"
OLD_FACTORS = PROJECT / "work/open_factor_v1/output/factor_values"
NEW = ROOT / "full_factors"
AUDIT = ROOT / "version10_dual_channel_cnn_xgb/audit"
OUT = ROOT / "version10_dual_channel_cnn_xgb/data"
OUT.mkdir(parents=True, exist_ok=True)

FIELDS = ["open", "high", "low", "close", "vwap_adj", "volume"]
FIELD_NAMES = ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "VOLUME"]


def panel_base(year: int) -> Path:
    return OLD_PANELS if year <= 2020 else NEW / f"year_{year}/panels"


def load_panels(year: int) -> dict[str, pd.DataFrame]:
    base = panel_base(year)
    frames = {name: pd.read_parquet(base / f"{name}.parquet") for name in FIELDS}
    # Add only the next year's first trading days for labels/execution across
    # year-end. Historical features remain sourced from the current panel.
    if year < 2026:
        next_base = panel_base(year + 1)
        for name in FIELDS:
            if frames[name].index.max().year <= year:
                nxt = pd.read_parquet(next_base / f"{name}.parquet")
                nxt = nxt.loc[(nxt.index.year == year + 1) &
                              (nxt.index <= pd.Timestamp(year + 1, 1, 20))]
                frames[name] = pd.concat([frames[name], nxt]).sort_index()
                frames[name] = frames[name].loc[~frames[name].index.duplicated(keep="last")]
    return frames


def eligible_rows(year: int, decisions: pd.DatetimeIndex) -> pd.DataFrame:
    raw = ds.dataset(RAW / "stock_market/stock_eod.parquet")
    table = raw.to_table(
        columns=["date", "code", "is_st", "is_suspend", "is_limit", "amount",
                 "volume", "float_cap", "zz500_wgt"],
        filter=(ds.field("date") >= f"{year}-01-01") &
               (ds.field("date") <= f"{year}-12-31"),
    ).to_pandas()
    table.date = pd.to_datetime(table.date)
    table = table.loc[
        table.date.isin(decisions) &
        table.code.str.match(r"^(60|68)\d{4}\.SH$|^(00|30)\d{4}\.SZ$")
    ].copy()
    ipo = pd.read_parquet(RAW / "stock_basic/key_dates.parquet", columns=["code", "ipo_date"])
    ipo.ipo_date = pd.to_datetime(ipo.ipo_date)
    table = table.merge(ipo.drop_duplicates("code"), on="code", how="left", validate="many_to_one")
    ok = (
        table.is_st.eq(0) & table.is_suspend.eq(0) & table.is_limit.eq(0) &
        table.amount.gt(0) & table.volume.gt(0) & table.float_cap.gt(0) &
        (table.date - table.ipo_date).dt.days.ge(180)
    )
    return table.loc[ok, ["date", "code", "float_cap", "zz500_wgt"]].sort_values(["date", "code"])


def factor_path(year: int, library: str, factor: str) -> Path:
    if year <= 2020:
        return OLD_FACTORS / library / f"{factor}.parquet"
    return NEW / f"year_{year}/factors/{library}/{factor}.parquet"


def read_factor_year(path: Path, year: int, decisions: pd.DatetimeIndex) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(
            path,
            filters=[("__index_level_0__", ">=", pd.Timestamp(year, 1, 1)),
                     ("__index_level_0__", "<=", pd.Timestamp(year, 12, 31))],
        )
    except Exception:
        frame = pd.read_parquet(path)
    return frame.reindex(index=decisions)


def index_series(start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.Series, pd.Series]:
    idx = ds.dataset(RAW / "index_market/index_eod.parquet").to_table(
        columns=["date", "code", "open", "close"],
        filter=(ds.field("code") == "000905.SH") &
               (ds.field("date") >= str(start.date())) &
               (ds.field("date") <= str(end.date())),
    ).to_pandas()
    idx.date = pd.to_datetime(idx.date)
    idx = idx.drop_duplicates("date").set_index("date").sort_index()
    return idx.close, idx.open


def build_year(year: int) -> None:
    print(f"[{year}] loading panels", flush=True)
    panels = load_panels(year)
    calendar = panels["close"].index.sort_values().unique()
    year_days = calendar[calendar.year == year]
    decisions = pd.DatetimeIndex(
        pd.Series(year_days, index=year_days).groupby(year_days.to_period("W-FRI")).max().to_numpy()
    )
    universe = eligible_rows(year, decisions)
    keys = pd.MultiIndex.from_frame(universe[["date", "code"]])
    csi_close, csi_open = index_series(calendar.min(), calendar.max())

    metadata, tensor_parts = [], []
    for date, group in universe.groupby("date", sort=True):
        loc = int(calendar.get_loc(date))
        if loc < 59:
            continue
        label_end = calendar[loc + 5] if loc + 5 < len(calendar) else pd.NaT
        entry_open = calendar[loc + 1] if loc + 1 < len(calendar) else pd.NaT
        exit_open = calendar[loc + 6] if loc + 6 < len(calendar) else pd.NaT
        codes = pd.Index(group.code)
        current_close = panels["close"].loc[date].reindex(codes).to_numpy(float)
        current_volume = panels["volume"].loc[date].reindex(codes).to_numpy(float)
        history_dates = calendar[loc - 59:loc + 1]
        channels = []
        for field in FIELDS[:5]:
            values = panels[field].reindex(index=history_dates, columns=codes).to_numpy(float).T
            channels.append((values / current_close[:, None] - 1.0) * 100.0)
        volume = panels["volume"].reindex(index=history_dates, columns=codes).to_numpy(float).T
        channels.append(volume / (current_volume[:, None] + 1.0))
        tensor = np.stack(channels, axis=2).astype("float32")
        # Exact source formula makes CLOSE00 constant zero. Record and enforce
        # only tiny floating-point differences, not missing observations.
        exact = np.isfinite(current_close)
        tensor[exact, -1, 3] = 0.0
        tensor_parts.append(tensor)

        target_stock = np.full(len(codes), np.nan)
        target_bench = np.nan
        execution = np.full(len(codes), np.nan)
        execution_bench = np.nan
        if pd.notna(label_end) and date in csi_close.index and label_end in csi_close.index:
            end_close = panels["close"].loc[label_end].reindex(codes).to_numpy(float)
            target_stock = end_close / current_close - 1.0
            bad = (~np.isfinite(target_stock)) | (target_stock < -0.99) | (target_stock > 3.0)
            target_stock[bad] = np.nan
            target_bench = float(csi_close.loc[label_end] / csi_close.loc[date] - 1.0)
        if (pd.notna(entry_open) and pd.notna(exit_open) and
                entry_open in csi_open.index and exit_open in csi_open.index):
            p0 = panels["open"].loc[entry_open].reindex(codes).to_numpy(float)
            p1 = panels["open"].loc[exit_open].reindex(codes).to_numpy(float)
            execution = p1 / p0 - 1.0
            bad = (~np.isfinite(execution)) | (execution < -0.99) | (execution > 3.0)
            execution[bad] = np.nan
            execution_bench = float(csi_open.loc[exit_open] / csi_open.loc[entry_open] - 1.0)
        m = group.copy()
        m["label_end_date"] = label_end
        m["entry_open"] = entry_open
        m["exit_open"] = exit_open
        m["target_stock_5d_close"] = target_stock
        m["target_csi500_5d_close"] = target_bench
        m["target_relative_5d"] = target_stock - target_bench
        m["execution_stock_open"] = execution
        m["execution_csi500_open"] = execution_bench
        m["cnn_finite_fraction"] = np.isfinite(tensor).mean(axis=(1, 2)).astype("float32")
        metadata.append(m)
    meta = pd.concat(metadata, ignore_index=True)
    cnn = np.concatenate(tensor_parts, axis=0)
    assert len(meta) == len(cnn)

    candidates = pd.read_csv(AUDIT / "xgboost_candidates_non_alpha360_96.csv")
    xgb = meta[["date", "code"]].copy().set_index(["date", "code"])
    all_keys = xgb.index
    for j, row in enumerate(candidates.itertuples(index=False), 1):
        frame = read_factor_year(factor_path(year, row.library, row.factor), year, decisions)
        values = frame.stack(future_stack=True).reindex(all_keys)
        # Comparable scale and robust tails, fitted without future returns.
        xgb[row.因子ID] = values.groupby(level=0).rank(pct=True).astype("float32") - 0.5
        if j % 20 == 0:
            print(f"[{year}] XGB factors {j}/96", flush=True)
    xgb = xgb.reset_index()
    assert (xgb[["date", "code"]].to_numpy() == meta[["date", "code"]].to_numpy()).all()
    output = meta.merge(xgb, on=["date", "code"], how="left", validate="one_to_one")
    output.to_parquet(OUT / f"{year}_common_xgb.parquet", index=False)
    np.save(OUT / f"{year}_cnn.npy", cnn)
    summary = {
        "year": year,
        "rows": len(output),
        "dates": int(output.date.nunique()),
        "first_date": str(output.date.min().date()),
        "last_date": str(output.date.max().date()),
        "labeled_rows": int(output.target_relative_5d.notna().sum()),
        "fully_labeled_dates": int(output.groupby("date").target_relative_5d.apply(lambda x: x.notna().any()).sum()),
        "cnn_shape": list(cnn.shape),
        "cnn_finite_share": float(np.isfinite(cnn).mean()),
        "xgb_features": len(candidates),
        "xgb_finite_share": float(output[candidates.因子ID].notna().mean().mean()),
        "label": "close(t)->close(t+5 trading days) stock return minus official 000905.SH",
        "execution": "open(t+1)->open(t+6), never used to prefilter prediction universe",
    }
    (OUT / f"{year}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[{year}] saved", summary, flush=True)
    del panels, meta, cnn, xgb, output


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=list(range(2015, 2027)))
    args = parser.parse_args()
    for y in args.years:
        build_year(y)
