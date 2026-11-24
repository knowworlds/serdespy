#!/usr/bin/env python3
"""SerDes error attribution from ADC codes: simulate, fit, rank, plot.

    python src/112g_pam4/attribution/run_attribution.py
    python src/112g_pam4/attribution/run_attribution.py --media out/ --mc 10

The simulator knows the truth (lane mismatches, THD, jitter, noise). The analyzer
sees only the ADC codes of four captures plus the known channel and receiver
settings of each capture. It prints the ranking, the parameter table with
z = (estimate − truth) / standard error, and writes the plots.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from budget import (  # noqa: E402
    LABEL,
    SOURCES,
    Budget,
    ablation_truth,
    estimated_budget,
    identifiability,
    lane_report,
    measured_mse,
    parameter_table,
)
from chain import (  # noqa: E402
    CODE_MAX,
    CODE_MIN,
    EA,
    LSB,
    build_receiver,
    default_captures,
    default_truth,
    simulate,
)
from estimator import fit_mean, fit_noise, prepare  # noqa: E402

plt.rcParams["font.family"] = ["WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#17212e"
MUTED = "#55637a"
GRID = "#d9e0e9"
PURPLE = "#6d4bc4"
PURPLE_LIGHT = "#cbbdf0"
GREEN = "#2e9e6a"
ORANGE = "#d9822b"


def run(n_samples: int, seed_offset: int = 0, quiet: bool = False):
    truth = default_truth()
    settings = default_captures(n_samples)
    t_start = time.perf_counter()
    rxs, caps, fits, syncs = [], [], [], []
    for s in settings:
        s.seed += seed_offset
        rx = build_receiver(s)
        cap = simulate(rx, truth, np.random.default_rng(s.seed))
        rxs.append(rx)
        caps.append(cap)
    t_sim = time.perf_counter()
    for rx, cap in zip(rxs, caps):
        f, sy = prepare(rx, cap.codes)
        fits.append(f)
        syncs.append(sy)
    mf = fit_mean(fits)
    nf = fit_noise(fits)
    ident = identifiability(mf, nf)
    est, se, gains = estimated_budget(mf, nf, op=0)
    t_fit = time.perf_counter()
    tru = ablation_truth(rxs[0], truth, caps[0].truth_pj_phase)
    keys = [k for k, _ in SOURCES]
    flagged = set()
    for a in ident.aliases:
        flagged.update(g for g in a["groups"] if g in keys)
    if abs(ident.noise_corr_single[0]) > 0.9:
        flagged.update({"rx_noise", "adc_noise"})
    bud = Budget(
        keys,
        np.array([est[k] for k in keys]),
        np.array([se[k] for k in keys]),
        np.array([tru[k] for k in keys]),
        flagged,
        measured_mse(fits[0]),
    )
    lanes = lane_report(mf, truth)
    table = parameter_table(mf, nf, truth, lanes)
    timing = dict(simulate_s=t_sim - t_start, analyze_s=t_fit - t_sim)
    return dict(truth=truth, settings=settings, caps=caps, fits=fits, syncs=syncs, mf=mf, nf=nf,
                ident=ident, budget=bud, lanes=lanes, table=table, gains=gains, timing=timing)


def lane_z(lanes: dict) -> np.ndarray:
    return np.concatenate([(L["est"] - L["truth"]) / L["se"] for L in lanes.values()])


def report(r) -> dict:
    bud: Budget = r["budget"]
    sy = r["syncs"][0]
    share_e, share_t, share_se = bud.share("est"), bud.share("truth"), bud.share_se()
    order = np.argsort(share_e)[::-1]
    order_t = np.argsort(share_t)[::-1]
    ident = r["ident"]
    n_alias = len(ident.aliases) + (1 if abs(ident.noise_corr_single[0]) > 0.9 else 0)
    meas_snr = 10 * np.log10(EA / bud.measured_mse)
    print("=" * 78)
    print(f"同步 (仅凭 ADC 码): PRBS13Q {sy.mapping} · 递推通过率 {sy.recurrence_pass[sy.mapping]:.4f} · "
          f"起点 m={sy.m_start} · 线性解释 {100 * sy.linear_r2:.1f}%")
    print(f"均衡后 SNR (实测) {meas_snr:.2f} dB · 预算合计 {bud.snr_db('est'):.2f} dB · 消融真值合计 {bud.snr_db('truth'):.2f} dB")
    top = bud.keys[order[0]]
    print(f"最大误差来源: {LABEL[top]} {share_e[order[0]]:.1f}%  (真值 {share_t[order[0]]:.1f}%)")
    print(f"联合拟合 · {len(r['fits'])} 次采集 · 单次采集 A 的混淆方向 {n_alias} 个 (联合后全部可分)")
    print(f"计时: 仿真 {r['timing']['simulate_s']:.2f} s · 分析 {r['timing']['analyze_s']:.2f} s")
    print("-" * 78)
    print(f"{'#':>2} {'来源':<26}{'估计 %':>9}{'± se':>8}{'真值 %':>9}{'真值名次':>8}  flags")
    for rank, i in enumerate(order):
        sig = bud.est[i] > 2 * bud.est_se[i] or bud.est_se[i] == 0
        flags = ("" if sig else "不显著 ") + ("△" if bud.keys[i] in bud.flagged else "")
        print(f"{rank + 1:>2} {LABEL[bud.keys[i]]:<26}{share_e[i]:>9.2f}{share_se[i]:>8.2f}{share_t[i]:>9.2f}"
              f"{int(np.nonzero(order_t == i)[0][0]) + 1:>8}  {flags}")
    print("-" * 78)
    print(f"{'参数':<22}{'估计':>12}{'± se':>12}{'真值':>12}{'z':>7}")
    for row in r["table"]:
        fmt = row["fmt"]
        s = "—" if np.isnan(row["se"]) else fmt.format(row["se"])
        z = "—" if np.isnan(row["z"]) else f"{row['z']:+.2f}"
        print(f"{row['label']:<22}{fmt.format(row['est']):>12}{s:>12}{fmt.format(row['truth']):>12}{z:>7}  {row['unit']}")
    zl = lane_z(r["lanes"])
    zt = np.array([row["z"] for row in r["table"] if not np.isnan(row["z"])])
    print("-" * 78)
    print(f"逐路 z ({zl.size} 个): rms {np.sqrt(np.mean(zl**2)):.2f} · |z|<2 占 {np.mean(np.abs(zl) < 2) * 100:.0f}% · max |z| {np.max(np.abs(zl)):.2f}")
    print(f"参数表 z ({zt.size} 个): rms {np.sqrt(np.mean(zt**2)):.2f} · |z|<2 占 {np.mean(np.abs(zt) < 2) * 100:.0f}%")
    print(f"排名一致: 前8 顺序相同 = {list(order[:8]) == list(order_t[:8])} · 第一名相同 = {order[0] == order_t[0]}")
    print("混淆 (单次采集 A, 归一化 Fisher 特征值 < 1e-3):")
    for a in ident.aliases:
        print(f"  λ={a['eig']:.2e}  组: {', '.join(a['groups'])}  主要载荷: "
              + ", ".join(f"{n} {w:.2f}" for n, w in a["loadings"][:4]))
    print(f"  噪声: 单次采集 A 中 RX 噪声与 ADC 噪声估计相关系数 {ident.noise_corr_single[0]:+.3f}, 联合 {ident.noise_corr_joint:+.3f}")
    print(f"  联合 Fisher 最小特征值 {ident.eig_joint.min():.2e}; 单次: "
          + ", ".join(f"{s.name.split()[0]} {ident.eig_single[c].min():.1e}" for c, s in enumerate(r["settings"])))
    return dict(
        sync=dict(mapping=sy.mapping, recurrence_pass=sy.recurrence_pass, m_start=sy.m_start,
                  linear_r2=sy.linear_r2, symbol_errors=[s.symbol_errors for s in r["syncs"]]),
        snr_db=dict(measured=float(meas_snr), budget=bud.snr_db("est"), truth=bud.snr_db("truth")),
        ranking=[dict(key=bud.keys[i], label=LABEL[bud.keys[i]], est_pct=float(share_e[i]), se_pct=float(share_se[i]),
                      truth_pct=float(share_t[i]), flagged=bud.keys[i] in bud.flagged,
                      significant=bool(bud.est[i] > 2 * bud.est_se[i] or bud.est_se[i] == 0)) for i in order],
        params=[{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in row.items()} for row in r["table"]],
        lanes={k: dict(est=L["est"].tolist(), se=L["se"].tolist(), truth=L["truth"].tolist(), unit=L["unit"]) for k, L in r["lanes"].items()},
        lane_z=dict(rms=float(np.sqrt(np.mean(zl**2))), frac_lt2=float(np.mean(np.abs(zl) < 2)), max=float(np.max(np.abs(zl)))),
        aliases=[dict(eig=a["eig"], groups=a["groups"]) for a in ident.aliases],
        noise_corr=dict(single=ident.noise_corr_single, joint=ident.noise_corr_joint),
        pj_freq=r["mf"].pj_freq,
        timing=r["timing"],
    )


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def plot_ranking(r, path: Path):
    bud: Budget = r["budget"]
    se_pct = bud.share_se()
    est_pct, tru_pct = bud.share("est"), bud.share("truth")
    order = np.argsort(est_pct)[::-1][:8]
    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=130)
    y = np.arange(order.size)[::-1]
    for yi, i in zip(y, order):
        sig = bud.est[i] > 2 * bud.est_se[i] or bud.est_se[i] == 0
        ax.barh(yi, est_pct[i], height=0.62, color=PURPLE if sig else PURPLE_LIGHT)
        if se_pct[i] > 0:
            ax.errorbar(est_pct[i], yi, xerr=se_pct[i], color=INK, lw=0.9, capsize=2.5)
        ax.plot([tru_pct[i]] * 2, [yi - 0.38, yi + 0.38], color="black", lw=2)
        tri = " △" if bud.keys[i] in bud.flagged else ""
        ax.text(max(est_pct[i], tru_pct[i]) + se_pct[i] + 0.8, yi, f"{est_pct[i]:.1f}{tri}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[bud.keys[i]] for i in order], fontsize=9, color=INK)
    ax.set_xlabel("占总误差 %", color=MUTED, fontsize=9)
    ax.set_xlim(0, max(est_pct.max(), tru_pct.max()) * 1.15)
    ax.set_title("误差来源排名 (前8)", loc="left", fontsize=12, color=INK, pad=22)
    ax.text(0, 1.02, "紫 = 由 ADC 码估计 ± 标准误差 · 黑竖线 = 仿真消融真值 · 浅色 = 不显著 · △ = 单次采集有混淆",
            transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_lanes(r, path: Path):
    lanes = r["lanes"]
    spec = [
        ("adc_off", "ADC 偏移 (LSB)", GREEN), ("adc_gain", "ADC 增益 (%)", GREEN), ("adc_skew", "ADC 时序 (mUI)", GREEN),
        ("dac_off", "DAC 偏移 (%)", ORANGE), ("dac_gain", "DAC 增益 (%)", ORANGE), ("dac_skew", "DAC 时序 (mUI)", ORANGE),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.6), dpi=130)
    for ax, (key, title, color) in zip(axes.flat, spec):
        L = lanes[key]
        x = np.arange(L["est"].size)
        ax.bar(x, L["est"], width=0.62, color=color, alpha=0.85)
        ax.errorbar(x, L["est"], yerr=L["se"], fmt="none", ecolor=INK, lw=0.9, capsize=2.5)
        ax.hlines(L["truth"], x - 0.36, x + 0.36, color="black", lw=2)
        ax.axhline(0, color=GRID, lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([str(i) for i in x])
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        _style(ax)
    fig.suptitle("交织失配 (逐路)", x=0.01, ha="left", fontsize=12, color=INK)
    fig.text(0.01, 0.925, "绿 = ADC (按采样序号 n mod 8) · 橙 = DAC (按符号序号 m mod 4, 经信道记忆才能与 ADC 区分) · "
             "误差棒 = 标准误差 · 黑线 = 仿真真值", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_params(r, path: Path):
    rows = r["table"]
    cells, colors = [], []
    for row in rows:
        fmt = row["fmt"]
        s = "—" if np.isnan(row["se"]) else fmt.format(row["se"])
        z = "—" if np.isnan(row["z"]) else f"{row['z']:+.2f}"
        cells.append([row["section"], row["label"], f"{fmt.format(row['est'])} ± {s}", fmt.format(row["truth"]), row["unit"], z])
        bad = not np.isnan(row["z"]) and abs(row["z"]) >= 2
        colors.append(["white"] * 5 + ["#fde2e1" if bad else "#e5f4ec"])
    fig, ax = plt.subplots(figsize=(10, 0.3 * len(rows) + 0.6), dpi=130)
    ax.axis("off")
    tab = ax.table(cellText=cells, colLabels=["", "参数", "估计 ± 标准误差", "真值", "单位", "z"],
                   cellColours=colors, loc="center", cellLoc="left", colWidths=[0.06, 0.26, 0.26, 0.12, 0.12, 0.08])
    tab.auto_set_font_size(False)
    tab.set_fontsize(9)
    tab.scale(1, 1.25)
    for (i, _), cell in tab.get_celld().items():
        cell.set_edgecolor(GRID)
        if i == 0:
            cell.set_text_props(color=MUTED)
    ax.set_title("参数估计 (z = (估计 − 真值) / 标准误差)", loc="left", fontsize=12, color=INK)
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def plot_code_density(r, path: Path):
    f = r["fits"][0]
    nf = r["nf"]
    codes = np.round(f.x / LSB).astype(int)
    edges = np.arange(CODE_MIN, CODE_MAX + 2) - 0.5
    meas, _ = np.histogram(codes, edges)
    from estimator import _rj_feature as rjf

    var = nf.s_rx * f.rx.rx_acov[0] * f.jac["gain"] ** 2 + nf.s_adc + nf.s_rj * f.jac["gain"] ** 2 * rjf(f.jac, 0)
    rng = np.random.default_rng(5)
    reps = 16
    sim = f.jac["x"][None, :] + rng.standard_normal((reps, f.x.size)) * np.sqrt(np.maximum(var, 0))[None, :]
    model, _ = np.histogram(np.clip(np.round(sim / LSB), CODE_MIN, CODE_MAX), edges)
    model = model / reps
    centres = np.arange(CODE_MIN, CODE_MAX + 1)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 4.8), dpi=130, sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
    ax.bar(centres, meas, width=1.0, color="#9fb7d6", label="实测 ADC 码 (采集 A)")
    ax.plot(centres, model, color=PURPLE, lw=1.2, label="模型: 拟合均值 + 估计噪声")
    ax.legend(fontsize=8, frameon=False)
    ax.set_ylabel("计数", fontsize=9, color=MUTED)
    ax.set_title("码密度", loc="left", fontsize=12, color=INK)
    _style(ax)
    z = (meas - model) / np.sqrt(np.maximum(model, 1))
    ax2.bar(centres, z, width=1.0, color=MUTED)
    ax2.axhline(2, color=GRID, lw=0.8)
    ax2.axhline(-2, color=GRID, lw=0.8)
    ax2.set_ylabel("残差 / √N", fontsize=9, color=MUTED)
    ax2.set_xlabel("ADC 码", fontsize=9, color=MUTED)
    ax2.set_xlim(-100, 100)
    _style(ax2)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_identifiability(r, path: Path):
    ident = r["ident"]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), dpi=130)
    colors = ["#2b6cb0", "#2e9e6a", "#d9822b", "#b83280"]
    for c, s in enumerate(r["settings"]):
        w = np.sort(ident.eig_single[c])
        ax.semilogy(np.arange(w.size), np.maximum(w, 1e-12), "o-", ms=3, lw=1, color=colors[c], label=f"单次采集 {s.name}")
    wj = np.sort(ident.eig_joint)
    ax.semilogy(np.arange(wj.size), wj, "k.-", ms=3, lw=1.2, label="联合 4 次采集")
    ax.axhline(1e-3, color="#c53030", lw=0.8, ls="--")
    ax.text(0.5, 1.3e-3, "混淆阈值 1e-3", color="#c53030", fontsize=8)
    ax.set_xlabel("特征值序号 (升序)", fontsize=9, color=MUTED)
    ax.set_ylabel("归一化 Fisher 信息特征值", fontsize=9, color=MUTED)
    ax.set_title("可辨识性: 单次采集不可分, 联合拟合可分", loc="left", fontsize=12, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def plot_mc(z_all: np.ndarray, rank_ok: list, path: Path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=130)
    ax.hist(z_all, bins=np.linspace(-4, 4, 33), density=True, color="#9fb7d6", label=f"逐路 z, {z_all.size} 个")
    xs = np.linspace(-4, 4, 200)
    ax.plot(xs, np.exp(-xs**2 / 2) / np.sqrt(2 * np.pi), color=INK, lw=1.2, label="N(0, 1)")
    ax.set_title(f"蒙特卡洛 {len(rank_ok)} 次: z 的分布 (std {z_all.std():.2f}) · 第一名一致 {sum(rank_ok)}/{len(rank_ok)}",
                 loc="left", fontsize=10, color=INK)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", type=int, default=16384, help="ADC samples per capture")
    p.add_argument("--media", type=Path, default=HERE / "out", help="directory for plots and results.json")
    p.add_argument("--mc", type=int, default=0, help="extra Monte Carlo trials to check z calibration")
    args = p.parse_args(argv)
    args.media.mkdir(parents=True, exist_ok=True)
    r = run(args.samples)
    summary = report(r)
    paths = {
        "ranking": args.media / "attribution_ranking.png",
        "lanes": args.media / "attribution_lanes.png",
        "params": args.media / "attribution_params.png",
        "code_density": args.media / "attribution_code_density.png",
        "identifiability": args.media / "attribution_identifiability.png",
    }
    plot_ranking(r, paths["ranking"])
    plot_lanes(r, paths["lanes"])
    plot_params(r, paths["params"])
    plot_code_density(r, paths["code_density"])
    plot_identifiability(r, paths["identifiability"])
    if args.mc:
        zs, top_ok, order_ok, table_z = [], [], [], []
        for k in range(1, args.mc + 1):
            rk = run(args.samples, seed_offset=100 * k)
            zs.append(lane_z(rk["lanes"]))
            table_z.append([row["z"] for row in rk["table"] if not np.isnan(row["z"])])
            b = rk["budget"]
            oe, ot = np.argsort(b.est)[::-1], np.argsort(b.truth)[::-1]
            top_ok.append(bool(oe[0] == ot[0]))
            order_ok.append(int(np.sum(oe[:8] == ot[:8])))
            print(f"MC {k}: lane z rms {np.sqrt(np.mean(zs[-1] ** 2)):.2f} · 前8 同位 {order_ok[-1]}/8")
        z_all = np.concatenate(zs)
        tz = np.array(table_z)
        paths["mc"] = args.media / "attribution_mc_z.png"
        plot_mc(z_all, top_ok, paths["mc"])
        summary["mc"] = dict(trials=args.mc, lane_z_std=float(z_all.std()), lane_frac_lt2=float(np.mean(np.abs(z_all) < 2)),
                             table_z_std=tz.std(axis=0).tolist(), table_z_mean=tz.mean(axis=0).tolist(),
                             top1_match=sum(top_ok), top8_positions_match=order_ok)
        print(f"MC 汇总: 逐路 z std {z_all.std():.2f}, |z|<2 占 {np.mean(np.abs(z_all) < 2) * 100:.0f}%, 第一名一致 {sum(top_ok)}/{args.mc}")
    summary["plots"] = {k: str(v) for k, v in paths.items()}
    (args.media / "attribution_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    for v in paths.values():
        print(f"wrote {v}")


if __name__ == "__main__":
    main()
