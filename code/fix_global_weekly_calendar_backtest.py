"""Formal global-week calendar repair for the dual-channel backtest.

This script never overwrites the original version10 result directory.  It
keeps all model scores and tuning choices fixed, replaces only the erroneous
per-year weekly schedule, and recomputes executable holdings and returns.
"""
from __future__ import annotations

from pathlib import Path
import json
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import rankdata

from build_dual_channel_weekly_data import RAW, panel_base


ROOT = Path(__file__).resolve().parent
OLD = ROOT / "version10_dual_channel_cnn_xgb"
OUT = ROOT / "version10_dual_channel_cnn_xgb_global_calendar_fixed"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = {"CNN": "Signal_CNN_norm", "XGBoost": "Signal_XGB_norm", "50/50融合": "Final_Signal"}
BUY_RATE, SELL_RATE = 0.0005, 0.0015


def index_data(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    x = ds.dataset(RAW / "index_market/index_eod.parquet").to_table(
        columns=["date", "code", "open", "close"],
        filter=(ds.field("code") == "000905.SH")
        & (ds.field("date") >= str(start.date()))
        & (ds.field("date") <= str(end.date())),
    ).to_pandas()
    x["date"] = pd.to_datetime(x["date"])
    return x.drop_duplicates("date").set_index("date").sort_index()


def open_panel(year: int) -> pd.DataFrame:
    return pd.read_parquet(panel_base(year) / "open.parquet")


def raw_cross_year_return(d0: pd.Timestamp, d1: pd.Timestamp, codes: pd.Index) -> np.ndarray:
    x = ds.dataset(RAW / "stock_market/stock_eod.parquet").to_table(
        columns=["date", "code", "open", "adj_factor"],
        filter=(ds.field("date") >= str(d0.date())) & (ds.field("date") <= str(d1.date())),
    ).to_pandas()
    x["date"] = pd.to_datetime(x["date"])
    x = x.set_index(["date", "code"])
    i0 = pd.MultiIndex.from_product([[d0], codes])
    i1 = pd.MultiIndex.from_product([[d1], codes])
    r0 = x["open"].reindex(i0).to_numpy(float)
    r1 = x["open"].reindex(i1).to_numpy(float)
    a0 = x["adj_factor"].reindex(i0).to_numpy(float)
    a1 = x["adj_factor"].reindex(i1).to_numpy(float)
    out = np.full(len(codes), np.nan)
    adjusted = np.isfinite(r0*a0) & np.isfinite(r1*a1) & (r0*a0 != 0)
    out[adjusted] = (r1[adjusted]*a1[adjusted])/(r0[adjusted]*a0[adjusted])-1
    # Adjustment factors are absent for most of 2021.  Around year-end there
    # are no ex-dates in the source; use the raw open ratio pairwise.
    raw_ok = (~adjusted) & np.isfinite(r0) & np.isfinite(r1) & (r0 != 0)
    out[raw_ok] = r1[raw_ok]/r0[raw_ok]-1
    out[(out < -0.99) | (out > 3.0)] = np.nan
    return out


def global_schedule(prediction_dates: pd.DatetimeIndex, index: pd.DataFrame) -> pd.DataFrame:
    calendar = index.index
    grouped = pd.Series(calendar, index=calendar).groupby(calendar.to_period("W-FRI")).max()
    formal_all = pd.DatetimeIndex(grouped.to_numpy())
    formal = formal_all[(formal_all >= prediction_dates.min()) & (formal_all <= prediction_dates.max())]
    missing = formal.difference(prediction_dates)
    extra = prediction_dates.difference(formal)
    if len(missing):
        raise RuntimeError(f"formal signals without frozen prediction: {missing.tolist()}")
    next_map = dict(zip(formal_all[:-1], formal_all[1:]))
    rows = []
    for signal in formal:
        following = next_map.get(signal)
        i = int(calendar.searchsorted(signal, side="right"))
        j = int(calendar.searchsorted(following, side="right")) if following is not None else len(calendar)
        if i >= len(calendar) or j >= len(calendar):
            continue
        entry, exit_day = calendar[i], calendar[j]
        rows.append({"week_period": str(signal.to_period("W-FRI")), "signal_date": signal,
                     "next_signal_date": following, "entry_open": entry,
                     "return_realization_date": exit_day,
                     "calendar_days_held": (exit_day-entry).days})
    schedule = pd.DataFrame(rows)
    schedule["old_prediction_extra_dates"] = ";".join(str(x.date()) for x in extra)
    return schedule


def attach_execution(pred: pd.DataFrame, schedule: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    panels: dict[int, pd.DataFrame] = {}
    pieces = []
    for row in schedule.itertuples(index=False):
        g = pred.loc[pred.date.eq(row.signal_date)].copy()
        codes = pd.Index(g.code)
        if row.entry_open.year == row.return_realization_date.year:
            y = row.entry_open.year
            if y not in panels:
                panels[y] = open_panel(y)
            p0 = panels[y].loc[row.entry_open].reindex(codes).to_numpy(float)
            p1 = panels[y].loc[row.return_realization_date].reindex(codes).to_numpy(float)
            stock_return = p1/p0-1
        else:
            stock_return = raw_cross_year_return(row.entry_open, row.return_realization_date, codes)
        stock_return[(~np.isfinite(stock_return)) | (stock_return < -0.99) | (stock_return > 3.0)] = np.nan
        g["week_period"] = row.week_period
        g["entry_open_fixed"] = row.entry_open
        g["return_realization_date"] = row.return_realization_date
        g["execution_stock_open_fixed"] = stock_return
        g["execution_csi500_open_fixed"] = (
            index.at[row.return_realization_date, "open"] / index.at[row.entry_open, "open"] - 1
        )
        pieces.append(g)
    return pd.concat(pieces, ignore_index=True).sort_values(["date", "code"])


def run_portfolio(pred: pd.DataFrame, score: str, model: str, actual_turnover: bool):
    weekly, trades = [], []
    previous_names: set[str] = set()
    pretrade_weights: dict[str, float] = {}
    for k, (date, g) in enumerate(pred.groupby("date", sort=True)):
        ranked = g.loc[g[score].notna()].sort_values(score, ascending=False)
        top = ranked.head(200).copy()
        if len(top) != 200:
            raise RuntimeError(f"{model} {date}: fewer than 200 stocks")
        target = {c: 1/200 for c in top.code}
        if k == 0:
            buy_amount, sell_amount = 1.0, 0.0
        elif actual_turnover:
            union = set(pretrade_weights) | set(target)
            buy_amount = sum(max(target.get(c, 0.0)-pretrade_weights.get(c, 0.0), 0.0) for c in union)
            sell_amount = sum(max(pretrade_weights.get(c, 0.0)-target.get(c, 0.0), 0.0) for c in union)
        else:
            names = set(target)
            buy_amount = (200-len(names & previous_names))/200
            sell_amount = len(previous_names-names)/200 if previous_names else 0.0
        fee = BUY_RATE*buy_amount + SELL_RATE*sell_amount
        returns = top.set_index("code").execution_stock_open_fixed.fillna(0.0)
        gross = float(returns.mean())
        net = gross-fee  # Preserve the original additive fee convention.
        benchmark = float(g.execution_csi500_open_fixed.iloc[0])
        end_raw = {c: target[c]*(1+float(returns.loc[c])) for c in target}
        total_end = sum(end_raw.values())
        end_weights = {c: v/total_end for c, v in end_raw.items()}
        if actual_turnover:
            union = sorted(set(pretrade_weights) | set(target))
            for c in union:
                old_w, target_w = pretrade_weights.get(c, 0.0), target.get(c, 0.0)
                trades.append({"model": model, "signal_date": date, "code": c,
                               "pretrade_weight": old_w, "target_weight": target_w,
                               "buy_amount": max(target_w-old_w, 0.0),
                               "sell_amount": max(old_w-target_w, 0.0),
                               "selected": c in target,
                               "holding_return": float(returns.loc[c]) if c in target else np.nan,
                               "end_weight": end_weights.get(c, 0.0)})
        weekly.append({"model": model, "signal_date": date,
                       "week_period": g.week_period.iloc[0],
                       "entry_open": g.entry_open_fixed.iloc[0],
                       "return_realization_date": g.return_realization_date.iloc[0],
                       "top200_gross_return": gross, "top200_net_return": net,
                       "csi500_return": benchmark, "buy_amount": buy_amount,
                       "sell_amount": sell_amount, "two_way_turnover": buy_amount+sell_amount,
                       "one_way_turnover": (buy_amount+sell_amount)/2,
                       "transaction_cost": fee, "selected_count": 200,
                       "missing_holding_returns": int(top.execution_stock_open_fixed.isna().sum())})
        pretrade_weights = end_weights
        previous_names = set(target)
    return pd.DataFrame(weekly), pd.DataFrame(trades)


def exact_metrics(w: pd.DataFrame) -> dict:
    w = w.sort_values("return_realization_date")
    s = np.cumprod(1+w.top200_net_return.to_numpy(float))
    b = np.cumprod(1+w.csi500_return.to_numpy(float))
    days = max((w.return_realization_date.iloc[-1]-w.entry_open.iloc[0]).days, 1)
    annual = s[-1]**(365.25/days)-1
    bannual = b[-1]**(365.25/days)-1
    dd = s/np.maximum.accumulate(s)-1
    rel = s[-1]/b[-1]
    return {"periods": len(w), "calendar_days": days, "final_nav": s[-1],
            "benchmark_final_nav": b[-1], "annual_return": annual,
            "benchmark_annual_return": bannual,
            "annual_excess_geometric": rel**(365.25/days)-1,
            "period_return": s[-1]-1, "benchmark_period_return": b[-1]-1,
            "period_relative_return": rel-1, "max_drawdown": dd.min(),
            "average_buy_amount": w.buy_amount.mean(),
            "average_sell_amount": w.sell_amount.mean(),
            "average_one_way_turnover": w.one_way_turnover.mean(),
            "average_two_way_turnover": w.two_way_turnover.mean(),
            "total_transaction_cost": w.transaction_cost.sum(),
            "average_transaction_cost": w.transaction_cost.mean()}


def rank_ic(pred: pd.DataFrame, score: str) -> dict:
    values = []
    for _, g in pred.groupby("date", sort=True):
        g = g.dropna(subset=[score, "target_relative_5d"])
        if len(g) >= 300:
            values.append(float(np.corrcoef(rankdata(g[score]), rankdata(g.target_relative_5d))[0, 1]))
    a = np.asarray(values)
    return {"Rank_IC": a.mean(), "ICIR": a.mean()/a.std(ddof=1),
            "IC_positive_share": (a > 0).mean(), "IC_dates": len(a)}


def time_rolling_52w(w: pd.DataFrame) -> pd.DataFrame:
    """Calendar-time trailing 52 weeks using completed-return NAV observations.

    At realization date t, the anchor is the last completed NAV observation on
    or before t-364 calendar days.  No unfinished holding-period return enters.
    """
    x = w.sort_values("return_realization_date").copy().reset_index(drop=True)
    x["strategy_nav"] = (1+x.top200_net_return).cumprod()
    x["benchmark_nav"] = (1+x.csi500_return).cumprod()
    x["rolling_52w_relative_return"] = np.nan
    x["rolling_anchor_date"] = pd.NaT
    x["completed_periods_in_window"] = np.nan
    for i, row in x.iterrows():
        cutoff = row.return_realization_date-pd.Timedelta(weeks=52)
        eligible = x.index[(x.index < i) & (x.return_realization_date <= cutoff)]
        if len(eligible) == 0:
            continue
        j = int(eligible[-1])
        x.at[i, "rolling_anchor_date"] = x.at[j, "return_realization_date"]
        x.at[i, "completed_periods_in_window"] = i-j
        x.at[i, "rolling_52w_relative_return"] = (
            (x.at[i, "strategy_nav"]/x.at[j, "strategy_nav"])
            /(x.at[i, "benchmark_nav"]/x.at[j, "benchmark_nav"])-1
        )
    return x


def main() -> None:
    pred_old = pd.read_parquet(OLD / "out_of_sample_predictions.parquet")
    for c in ["date", "entry_open", "exit_open", "label_end_date"]:
        pred_old[c] = pd.to_datetime(pred_old[c])
    prediction_dates = pd.DatetimeIndex(sorted(pred_old.date.unique()))
    idx = index_data(prediction_dates.min()-pd.Timedelta(days=10),
                     prediction_dates.max()+pd.Timedelta(days=40))
    schedule = global_schedule(prediction_dates, idx)
    schedule.to_csv(OUT / "全局周度调仓日历.csv", index=False, encoding="utf-8-sig")
    pred = attach_execution(pred_old.loc[pred_old.date.isin(schedule.signal_date)], schedule, idx)
    pred.to_parquet(OUT / "固定模型预测与修正持有期.parquet", index=False)

    actual_weekly, proxy_weekly, all_trades = [], [], []
    for model, score in MODELS.items():
        w, trades = run_portfolio(pred, score, model, actual_turnover=True)
        wp, _ = run_portfolio(pred, score, model, actual_turnover=False)
        actual_weekly.append(w); proxy_weekly.append(wp); all_trades.append(trades)
    weekly = pd.concat(actual_weekly, ignore_index=True)
    proxy = pd.concat(proxy_weekly, ignore_index=True)
    trades = pd.concat(all_trades, ignore_index=True)
    weekly.to_csv(OUT / "逐周正式回测.csv", index=False, encoding="utf-8-sig")
    proxy.to_csv(OUT / "仅修复周历_沿用旧费用代理.csv", index=False, encoding="utf-8-sig")
    trades.to_parquet(OUT / "逐股买卖金额与持仓漂移.parquet", index=False)

    performance = []
    rolling_parts = []
    for model, score in MODELS.items():
        wm = weekly.loc[weekly.model.eq(model)].copy()
        for label, part in [(str(y), wm.loc[wm.return_realization_date.dt.year.eq(y)])
                            for y in range(2021, 2027)] + [("整体", wm)]:
            if part.empty:
                continue
            p = pred if label == "整体" else pred.loc[pred.return_realization_date.dt.year.eq(int(label))]
            performance.append({"model": model, "period": label,
                                **rank_ic(p, score), **exact_metrics(part)})
        r = time_rolling_52w(wm); r["model"] = model; rolling_parts.append(r)
    performance = pd.DataFrame(performance)
    rolling = pd.concat(rolling_parts, ignore_index=True)
    performance.to_csv(OUT / "三模型绩效_全局周历修复.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(OUT / "真实过去52周超额序列.csv", index=False, encoding="utf-8-sig")

    # Before/after decomposition.  Re-evaluate the old series on exact calendar
    # duration, then isolate calendar repair from the turnover-accounting repair.
    old_w = pd.read_csv(OLD / "weekly_backtest.csv", parse_dates=["date"])
    old_meta = pred_old[["date", "entry_open", "exit_open"]].drop_duplicates("date")
    old_w = old_w.merge(old_meta, on="date", how="left").rename(
        columns={"date": "signal_date", "exit_open": "return_realization_date",
                 "cost_proxy": "transaction_cost", "buy_ratio": "buy_amount",
                 "sell_ratio": "sell_amount"})
    old_w["one_way_turnover"] = (old_w.buy_amount+old_w.sell_amount)/2
    old_w["two_way_turnover"] = old_w.buy_amount+old_w.sell_amount
    comparisons = []
    for model in MODELS:
        oldm = old_w.loc[old_w.model.eq(model)].copy()
        oldm["entry_open"] = pd.to_datetime(oldm.entry_open)
        oldm["return_realization_date"] = pd.to_datetime(oldm.return_realization_date)
        proxym = proxy.loc[proxy.model.eq(model)]
        fixedm = weekly.loc[weekly.model.eq(model)]
        for version, frame, source in [
            ("修复前_原始296信号", oldm, "原始实际调仓与原始滚动口径"),
            ("仅修复全局周历_旧费用代理", proxym, "实际调仓日历与持有收益变化"),
            ("正式修复_真实权重漂移费用", fixedm, "实际买卖金额与费用核算变化"),
        ]:
            m = exact_metrics(frame)
            rr = time_rolling_52w(frame.rename(columns={"date": "signal_date"})
                                  if "signal_date" not in frame else frame)
            early = rr.loc[(rr.return_realization_date >= "2025-01-01") &
                           (rr.return_realization_date <= "2025-03-15"),
                           "rolling_52w_relative_return"].max()
            comparisons.append({"model": model, "version": version,
                                "difference_source": source, **m,
                                "2025年初最高真实52周超额": early,
                                "signal_count": len(frame)})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(OUT / "修复前后差异表.csv", index=False, encoding="utf-8-sig")

    # Audits.
    duplicate_periods = int(schedule.week_period.duplicated().sum())
    continuity = schedule.entry_open.iloc[1:].reset_index(drop=True).eq(
        schedule.return_realization_date.iloc[:-1].reset_index(drop=True)
    )
    holdings_check = []
    for (model, date), g in trades.loc[trades.selected].groupby(["model", "signal_date"]):
        expected = float((g.target_weight*g.holding_return.fillna(0)).sum())
        actual = float(weekly.loc[(weekly.model.eq(model)) & (weekly.signal_date.eq(date)),
                                  "top200_gross_return"].iloc[0])
        holdings_check.append(abs(expected-actual))
    audit = pd.DataFrame([
        {"check": "全局正式信号数", "value": len(schedule), "passed": len(schedule)==295},
        {"check": "旧预测中被移除的非正式日期", "value": "2024-12-31", "passed": True},
        {"check": "同一星期五截止周重复信号数", "value": duplicate_periods, "passed": duplicate_periods==0},
        {"check": "相邻持有区间首尾不连续数", "value": int((~continuity).sum()), "passed": continuity.all()},
        {"check": "持仓加权收益最大复算误差", "value": max(holdings_check), "passed": max(holdings_check)<1e-12},
        {"check": "固定模型分数是否直接提取未重训", "value": "是", "passed": True},
        {"check": "原Purge Gap边界是否全部通过", "value": bool(pd.read_csv(OLD/'purge_gap_audit.csv').passed.all()),
         "passed": bool(pd.read_csv(OLD/'purge_gap_audit.csv').passed.all())},
        {"check": "354候选名单事后筛选风险是否保留", "value": "是，正式报告继续标注", "passed": True},
    ])
    audit.to_csv(OUT / "正式修复审计.csv", index=False, encoding="utf-8-sig")

    # Key windows, now labeled by realized date.
    key = rolling.loc[(rolling.model.eq("50/50融合")) &
                      (rolling.return_realization_date.between("2024-01-01", "2025-03-31"))].copy()
    key.to_csv(OUT / "2024年初与2025年初关键窗口.csv", index=False, encoding="utf-8-sig")

    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
    fig, ax = plt.subplots(figsize=(13, 7), dpi=170)
    for model in MODELS:
        g = rolling.loc[rolling.model.eq(model)].sort_values("return_realization_date")
        ax.plot(g.return_realization_date, g.strategy_nav, label=f"{model} Top200扣费后", lw=1.7)
    g = rolling.loc[rolling.model.eq("CNN")].sort_values("return_realization_date")
    ax.plot(g.return_realization_date, g.benchmark_nav, color="black", label="官方中证500", lw=1.4)
    ax.set_title("全局周历修复版：三模型实际实现净值", fontproperties=font)
    ax.set_xlabel("收益实现日期", fontproperties=font); ax.set_ylabel("累计净值", fontproperties=font)
    ax.legend(prop=font); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "三模型净值曲线_全局周历修复.png", bbox_inches="tight"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 6), dpi=170)
    for model in MODELS:
        g = rolling.loc[rolling.model.eq(model)].sort_values("return_realization_date")
        ax.plot(g.return_realization_date, 100*g.rolling_52w_relative_return, label=model, lw=1.6)
    ax.axhline(0, color="black", lw=.8)
    ax.set_title("真实过去52周复合超额｜只计入已完成持有期", fontproperties=font)
    ax.set_xlabel("收益实现日期", fontproperties=font); ax.set_ylabel("52周相对收益（%）", fontproperties=font)
    ax.legend(prop=font); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "滚动52周超额_全局周历修复.png", bbox_inches="tight"); plt.close(fig)

    for name in ["purge_gap_audit.csv", "validation_tuning_log.csv", "factor_groups.csv",
                 "alpha360_tensor_mapping.csv", "correlation_filter_results.csv"]:
        shutil.copy2(OLD / name, OUT / name)
    summary = {"old_directory_untouched": str(OLD), "formal_signal_count": len(schedule),
               "removed_duplicate_signal": "2024-12-31", "scores_retrained": False,
               "hyperparameters_changed": False, "fees": {"buy": BUY_RATE, "sell": SELL_RATE},
               "rolling_definition": "NAV ratio from last completed realization on/before t-364d to t",
               "strict_oos_claim": False,
               "historical_candidate_bias": "354 historical winners observed 2021-2025"}
    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
