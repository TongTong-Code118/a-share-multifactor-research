"""Frozen-score Top100 experiment with a 20% one-way weekly turnover cap."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import pandas as pd

from fix_global_weekly_calendar_backtest import exact_metrics, time_rolling_52w
from run_turnover_cap_sensitivity import MODELS, load_panel, run_cap


ROOT = Path(__file__).resolve().parent
BASE = ROOT / "version10_dual_channel_cnn_xgb_global_calendar_fixed"
TOP200 = BASE / "换手约束敏感性_10_15_20_30"
OUT = BASE / "Top100_单边换手20"
OUT.mkdir(parents=True, exist_ok=True)
CAP, TARGET_N = 0.20, 100


def pct(x):
    return f"{100*x:.2f}%"


def main():
    pred = pd.read_parquet(BASE / "固定模型预测与修正持有期.parquet")
    for c in ["date", "entry_open_fixed", "return_realization_date"]:
        pred[c] = pd.to_datetime(pred[c])
    cache, weekly_parts = {}, []
    for model, score in MODELS.items():
        print(model, flush=True)
        weekly_parts.append(run_cap(pred, score, model, CAP, cache, target_n=TARGET_N))
    weekly = pd.concat(weekly_parts, ignore_index=True)
    weekly.to_csv(OUT / "Top100_换手20_逐周回测.csv", index=False, encoding="utf-8-sig")

    perf, rolling_parts = [], []
    for model, w in weekly.groupby("model", sort=False):
        for label, part in [(str(y), w.loc[w.return_realization_date.dt.year.eq(y)])
                            for y in range(2021, 2027)] + [("整体", w)]:
            if part.empty:
                continue
            perf.append({"model": model, "period": label, **exact_metrics(part),
                         "cap_binding_share_ex_first": float(part.loc[~part.first_formation_exempt,
                                                                       "cap_binding"].mean()),
                         "average_target_top100_coverage": float(part.target_set_weight_coverage.mean()),
                         "average_actual_holding_count": float(part.actual_holding_count.mean()),
                         "average_max_single_stock_weight": float(part.max_holding_weight_after_trade.mean()),
                         "maximum_single_stock_weight": float(part.max_holding_weight_after_trade.max())})
        r = time_rolling_52w(w); r["model"] = model; rolling_parts.append(r)
    perf = pd.DataFrame(perf)
    rolling = pd.concat(rolling_parts, ignore_index=True)
    perf.to_csv(OUT / "Top100_换手20_绩效.csv", index=False, encoding="utf-8-sig")
    rolling.to_csv(OUT / "Top100_换手20_真实52周超额.csv", index=False, encoding="utf-8-sig")

    # Execution audit.
    audit = []
    for model, g in weekly.groupby("model"):
        x = g.loc[~g.first_formation_exempt]
        audit.append({"模型": model, "正式期数": len(g), "最少持仓数": g.actual_holding_count.min(),
                      "最多持仓数": g.actual_holding_count.max(),
                      "最大实际单边换手": x.executed_one_way_turnover.max(),
                      "超限次数": int((x.executed_one_way_turnover > CAP+1e-10).sum()),
                      "上限触发比例": x.cap_binding.mean(),
                      "平均Top100覆盖权重": g.target_set_weight_coverage.mean(),
                      "最大单票权重": g.max_holding_weight_after_trade.max()})
    pd.DataFrame(audit).to_csv(OUT / "Top100_换手20_执行审计.csv", index=False, encoding="utf-8-sig")

    # Top100 vs frozen Top200-20% comparison for the fusion model.
    p200 = pd.read_csv(TOP200 / "四档换手约束绩效.csv")
    p200 = p200.loc[(p200.model.eq("50/50融合")) & p200.turnover_cap.eq(CAP)].copy()
    p100 = perf.loc[perf.model.eq("50/50融合")].copy()
    compare = pd.concat([
        p200.assign(portfolio="Top200_换手20"), p100.assign(portfolio="Top100_换手20")
    ], ignore_index=True, sort=False)
    compare.to_csv(OUT / "Top100与Top200_换手20_对比.csv", index=False, encoding="utf-8-sig")

    r100 = rolling.loc[rolling.model.eq("50/50融合")].sort_values("return_realization_date")
    r200_all = pd.read_csv(TOP200 / "四档换手约束真实52周超额.csv",
                           parse_dates=["return_realization_date"])
    r200 = r200_all.loc[(r200_all.model.eq("50/50融合")) &
                        r200_all.turnover_cap.eq(CAP)].sort_values("return_realization_date")
    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc")
    fig, axes = plt.subplots(2, 1, figsize=(13, 10), dpi=170, sharex=True)
    for label, g, color in [("Top100｜换手20%", r100, "#D95F02"),
                            ("Top200｜换手20%", r200, "#2474B5")]:
        axes[0].plot(g.return_realization_date, g.strategy_nav/g.benchmark_nav,
                     label=label, color=color, lw=1.7)
        axes[1].plot(g.return_realization_date, 100*g.rolling_52w_relative_return,
                     label=label, color=color, lw=1.6)
    axes[0].axhline(1, color="black", lw=.8)
    axes[0].set_title("Top100与Top200：累计相对中证500净值", fontproperties=font)
    axes[0].set_ylabel("相对净值", fontproperties=font)
    axes[1].axhline(0, color="black", lw=.8)
    axes[1].set_title("真实过去52周超额｜单边换手上限20%", fontproperties=font)
    axes[1].set_ylabel("52周超额（%）", fontproperties=font)
    axes[1].set_xlabel("收益实现日期", fontproperties=font)
    for ax in axes:
        ax.legend(prop=font); ax.grid(alpha=.2); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "Top100与Top200_换手20_超额稳定性图.png", bbox_inches="tight"); plt.close(fig)

    f100 = p100.set_index("period"); f200 = p200.set_index("period")
    overall100, overall200 = f100.loc["整体"], f200.loc["整体"]
    years = "\n".join(
        f"| {y} | {pct(f100.loc[str(y),'period_return'])} | {pct(f200.loc[str(y),'period_return'])} | "
        f"{pct(f100.loc[str(y),'period_relative_return'])} | {pct(f200.loc[str(y),'period_relative_return'])} |"
        for y in range(2021, 2027)
    )
    report = f"""# Top100多头＋单边周换手20%实验

## 结论

Top100实验已经在修复后的295期全局周历上完成。模型、预测、训练验证、Purge Gap、费用率和交易时间全部与Top200换手20%版本一致，唯一核心变化是目标持仓数由200只缩小为100只。

融合模型Top100的年化净收益为 **{pct(overall100.annual_return)}**，几何年化超额为 **{pct(overall100.annual_excess_geometric)}**，最大回撤为 **{pct(overall100.max_drawdown)}**。对应Top200版本分别为 **{pct(overall200.annual_return)}**、**{pct(overall200.annual_excess_geometric)}** 和 **{pct(overall200.max_drawdown)}**。

## 1. 交易规则

- 每周将全市场股票按冻结融合信号排序，目标名单取前100只，初始目标权重每只1%。
- 首次建仓允许100%；之后单边实际换手不超过20%。
- 先卖出模型排名最弱的旧持仓，再买入排名最高的新股票，累计卖出金额接近20%即停止。
- 未完成的名单替换留到后续周；始终保持100只股票。
- 买入费率5bp，卖出费率15bp；持有期和收益实现日期沿用全局周历修复版。

## 2. 整体比较

| 组合 | 年化净收益 | 几何年化超额 | 最大回撤 | 平均单边换手 | 累计费用率之和 | 平均目标名单覆盖权重 |
|---|---:|---:|---:|---:|---:|---:|
| Top100＋20% | {pct(overall100.annual_return)} | {pct(overall100.annual_excess_geometric)} | {pct(overall100.max_drawdown)} | {pct(overall100.average_one_way_turnover)} | {pct(overall100.total_transaction_cost)} | {pct(overall100.average_target_top100_coverage)} |
| Top200＋20% | {pct(overall200.annual_return)} | {pct(overall200.annual_excess_geometric)} | {pct(overall200.max_drawdown)} | {pct(overall200.average_one_way_turnover)} | {pct(overall200.total_transaction_cost)} | {pct(overall200.average_target_top200_coverage)} |

## 3. 逐年结果

| 收益实现年度 | Top100收益 | Top200收益 | Top100相对指数 | Top200相对指数 |
|---|---:|---:|---:|---:|
{years}

## 4. 风险与执行审计

- 所有正式调仓期均保持100只股票，换手超限次数为0。
- Top100的平均最大单票权重为 **{pct(overall100.average_max_single_stock_weight)}**，全期观察到的最大单票权重为 **{pct(overall100.maximum_single_stock_weight)}**；由于保留持仓不强制恢复1%等权，权重会随价格自然漂移。
- 平均只有 **{pct(overall100.average_target_top100_coverage)}** 的仓位属于当期最新Top100，说明20%换手上限对更集中的Top100目标同样形成明显滞后。
- 本实验属于新增的测试期敏感性分析，不能因为其历史结果优于或劣于Top200就反向修改原模型超参数。

## 5. 文件

- `Top100与Top200_换手20_超额稳定性图.png`
- `Top100_换手20_逐周回测.csv`
- `Top100_换手20_绩效.csv`
- `Top100_换手20_真实52周超额.csv`
- `Top100与Top200_换手20_对比.csv`
- `Top100_换手20_执行审计.csv`
"""
    (OUT / "Top100_换手20_实验总结.md").write_text(report, encoding="utf-8")
    print(report.split("## 1.")[0])


if __name__ == "__main__":
    main()
