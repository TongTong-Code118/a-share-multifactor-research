"""Run 10/15/20/30% one-way turnover caps on frozen dual-model scores."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
import pandas as pd

from build_dual_channel_weekly_data import panel_base
from fix_global_weekly_calendar_backtest import (
    BUY_RATE, SELL_RATE, exact_metrics, raw_cross_year_return, time_rolling_52w,
)


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "version10_dual_channel_cnn_xgb_global_calendar_fixed"
OUT = BASE / "换手约束敏感性_10_15_20_30"
OUT.mkdir(parents=True, exist_ok=True)
CAPS = [0.10, 0.15, 0.20, 0.30]
MODELS = {"CNN": "Signal_CNN_norm", "XGBoost": "Signal_XGB_norm", "50/50融合": "Final_Signal"}


def load_panel(year, cache):
    if year not in cache:
        cache[year] = pd.read_parquet(panel_base(year) / "open.parquet")
    return cache[year]


def complete_returns(g, codes, cache):
    known = g.set_index("code").execution_stock_open_fixed
    result = known.reindex(codes)
    missing = result.index[result.isna()]
    if len(missing):
        entry = pd.Timestamp(g.entry_open_fixed.iloc[0])
        exit_day = pd.Timestamp(g.return_realization_date.iloc[0])
        if entry.year == exit_day.year:
            p = load_panel(entry.year, cache)
            p0 = p.loc[entry].reindex(missing).to_numpy(float)
            p1 = p.loc[exit_day].reindex(missing).to_numpy(float)
            extra = p1/p0-1
        else:
            extra = raw_cross_year_return(entry, exit_day, missing)
        result.loc[missing] = extra
    return result.fillna(0.0)


def run_cap(pred, score, model, cap, cache, target_n=200):
    rows = []
    pretrade = {}
    for k, (date, g) in enumerate(pred.groupby("date", sort=True)):
        top = g.loc[g[score].notna()].nlargest(target_n, score)
        desired = {c: 1/target_n for c in top.code}
        if k == 0:
            post = desired.copy()
            required = executed = buy = 1.0
            sell = 0.0
            completion = 1.0
        else:
            score_map = g.set_index("code")[score]
            held = set(pretrade)
            desired_names = set(desired)
            # Full desired name replacement turnover, measured at drifted weights.
            exit_candidates = sorted(
                held-desired_names,
                key=lambda c: float(score_map.get(c, -np.inf))
                if pd.notna(score_map.get(c, np.nan)) else -np.inf,
            )
            entry_candidates = [c for c in top.code if c not in held]
            required = sum(pretrade[c] for c in exit_candidates)
            post = pretrade.copy()
            executed = 0.0
            entry_index = 0
            for old_name in exit_candidates:
                if entry_index >= len(entry_candidates):
                    break
                amount = pretrade[old_name]
                if executed+amount > cap+1e-12:
                    continue
                new_name = entry_candidates[entry_index]
                entry_index += 1
                del post[old_name]
                post[new_name] = amount
                executed += amount
            buy = sell = executed
            completion = executed/required if required > 1e-15 else 1.0
        fee = BUY_RATE*buy+SELL_RATE*sell
        codes = pd.Index(list(post))
        stock_return = complete_returns(g, codes, cache)
        gross = float(sum(post[c]*float(stock_return.loc[c]) for c in codes))
        net = gross-fee
        end_raw = {c: post[c]*(1+float(stock_return.loc[c])) for c in codes}
        end_total = sum(end_raw.values())
        pretrade = {c: w/end_total for c, w in end_raw.items() if w/end_total > 1e-12}
        target_coverage = sum(post.get(c, 0.0) for c in desired)
        rows.append({"model": model, "turnover_cap": cap, "target_count": target_n, "signal_date": date,
                     "entry_open": g.entry_open_fixed.iloc[0],
                     "return_realization_date": g.return_realization_date.iloc[0],
                     "gross_return": gross, "top200_net_return": net,
                     "csi500_return": float(g.execution_csi500_open_fixed.iloc[0]),
                     "required_one_way_turnover": required,
                     "executed_one_way_turnover": executed,
                     "buy_amount": buy, "sell_amount": sell,
                     "one_way_turnover": executed, "two_way_turnover": buy+sell,
                     "transaction_cost": fee,
                     "replacement_completion": completion,
                     "cap_binding": bool(k > 0 and required > cap+1e-12),
                     "actual_holding_count": len(post),
                     "max_holding_weight_after_trade": max(post.values()),
                     "min_holding_weight_after_trade": min(post.values()),
                     "target_set_weight_coverage": target_coverage,
                     "target_top200_weight_coverage": target_coverage,
                     "first_formation_exempt": k == 0})
    return pd.DataFrame(rows)


def main():
    pred = pd.read_parquet(BASE / "固定模型预测与修正持有期.parquet")
    for c in ["date", "entry_open_fixed", "return_realization_date"]:
        pred[c] = pd.to_datetime(pred[c])
    cache = {}
    weekly_parts = []
    for model, score in MODELS.items():
        for cap in CAPS:
            print(model, cap, flush=True)
            weekly_parts.append(run_cap(pred, score, model, cap, cache))
    weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly.to_csv(OUT / "四档换手约束逐周回测.csv", index=False, encoding="utf-8-sig")

    perf, rolling_parts = [], []
    for (model, cap), w in weekly.groupby(["model", "turnover_cap"], sort=False):
        for label, part in [(str(y), w.loc[w.return_realization_date.dt.year.eq(y)])
                            for y in range(2021, 2027)] + [("整体", w)]:
            if part.empty:
                continue
            perf.append({"model": model, "turnover_cap": cap, "period": label,
                         **exact_metrics(part),
                         "cap_binding_share_ex_first": float(part.loc[~part.first_formation_exempt,
                                                                       "cap_binding"].mean()),
                         "average_target_top200_coverage": float(part.target_top200_weight_coverage.mean()),
                         "average_actual_holding_count": float(part.actual_holding_count.mean())})
        r = time_rolling_52w(w); r["model"] = model; r["turnover_cap"] = cap
        rolling_parts.append(r)
    perf = pd.DataFrame(perf)
    rolling = pd.concat(rolling_parts, ignore_index=True)
    perf.to_csv(OUT / "四档换手约束绩效.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(OUT / "四档换手约束真实52周超额.csv", index=False, encoding="utf-8-sig")

    # Add the repaired unconstrained version as a benchmark in the summary.
    unc = pd.read_csv(BASE / "三模型绩效_全局周历修复.csv")
    unc = unc.loc[unc.period.eq("整体")].copy()
    comparison = perf.loc[perf.period.eq("整体")].copy()
    comparison.to_csv(OUT / "四档换手约束整体对比.csv", index=False, encoding="utf-8-sig")

    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
    fusion_roll = rolling.loc[rolling.model.eq("50/50融合")]
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), dpi=170, sharex=True)
    colors = {0.10: "#1f77b4", 0.15: "#2ca02c", 0.20: "#ff7f0e", 0.30: "#9467bd"}
    for cap in CAPS:
        g = fusion_roll.loc[fusion_roll.turnover_cap.eq(cap)].sort_values("return_realization_date")
        relative_nav = g.strategy_nav/g.benchmark_nav
        axes[0].plot(g.return_realization_date, relative_nav, label=f"单边上限{cap:.0%}",
                     lw=1.6, color=colors[cap])
        axes[1].plot(g.return_realization_date, 100*g.rolling_52w_relative_return,
                     label=f"单边上限{cap:.0%}", lw=1.5, color=colors[cap])
    axes[0].axhline(1, color="black", lw=.8)
    axes[0].set_title("融合模型：四档换手上限累计相对净值", fontproperties=font)
    axes[0].set_ylabel("相对中证500净值", fontproperties=font)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_title("融合模型：真实过去52周超额", fontproperties=font)
    axes[1].set_ylabel("52周超额（%）", fontproperties=font)
    axes[1].set_xlabel("收益实现日期", fontproperties=font)
    for ax in axes:
        ax.legend(prop=font, ncol=2); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "融合模型_四档换手上限_超额稳定性图.png", bbox_inches="tight"); plt.close(fig)

    summary = {"caps": CAPS, "first_formation_exempt": True,
               "target": "frozen weekly Top200 equal weight",
               "execution": "keep 200 names; replace lowest-score old holdings with highest-score entrants until the drifted-weight cap is reached",
               "model_scores_changed": False, "fees": {"buy": BUY_RATE, "sell": SELL_RATE},
               "test_based_parameter_selection": False}
    (OUT / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(comparison.loc[comparison.model.eq("50/50融合"),
                         ["turnover_cap", "annual_return", "annual_excess_geometric",
                          "max_drawdown", "average_one_way_turnover", "total_transaction_cost",
                          "average_target_top200_coverage"]].to_string(index=False))


if __name__ == "__main__":
    main()
