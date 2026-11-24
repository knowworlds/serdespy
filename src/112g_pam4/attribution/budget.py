"""Error-source budget at the operating capture, identifiability, and the parameter table.

Every source is expressed as its mean-square contribution to the slicer error
after the known 12-tap FFE (÷ f0), so shares add up to the equalized-SNR loss.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chain import (
    EA,
    LEVELS,
    LSB,
    Impairments,
    Params,
    Receiver,
    evaluate,
    ffe_symbol_index,
    make_layout,
    prbs_symbols,
    simulate,
    slicer_error,
)
from estimator import B4, B8, LOCAL, SHARED, MeanFit, NoiseFit

SOURCES = [
    ("rx_noise", "RX 输入噪声 (VGA 之前)"),
    ("dac_lane", "TX DAC 交织失配"),
    ("thd", "TX DAC 非线性 (THD)"),
    ("rj", "TX 边沿随机抖动"),
    ("isi", "残余 ISI (均衡器长度受限)"),
    ("adc_lane", "ADC 交织失配"),
    ("adc_noise", "ADC 热噪声"),
    ("pj", "周期抖动 PJ (CDR 后残余)"),
    ("quant", "ADC 量化噪声"),
]
LABEL = dict(SOURCES)
NS = len(SHARED)


def shared_columns(group: str) -> list[int]:
    from estimator import GROUP_OF

    return [i for i, (name, _) in enumerate(SHARED) if GROUP_OF.get(name) == group]


def theta_shared(p: Params) -> np.ndarray:
    return np.concatenate(
        [
            B8.T @ p.adc_off,
            B8.T @ p.adc_gain,
            B8.T @ p.adc_skew,
            B4.T @ p.dac_off,
            B4.T @ p.dac_gain,
            B4.T @ p.dac_skew,
            [p.a2, p.a3],
        ]
    )


def _ffe_cols(rx: Receiver, E: np.ndarray) -> np.ndarray:
    return np.column_stack([slicer_error(rx, E[:, i]) for i in range(E.shape[1])])


def _project_out(E: np.ndarray, N: np.ndarray) -> np.ndarray:
    coef, *_ = np.linalg.lstsq(N, E, rcond=None)
    return E - N @ coef


def rj_gain(rx: Receiver, jac: dict) -> float:
    """Mean FFE-output RJ variance per unit edge variance: E_j Σ_m d_m² (Σ_t w_t S'(t_{n−t} − m))² / f0²."""

    d = jac["d"][jac["idx"]]
    g = jac["g1"] * d * jac["gain"][:, None]
    n, k = g.shape
    taps = rx.w.size
    nz = n - taps + 1
    acc = np.zeros((nz, k + taps))
    for t in range(taps):
        rows = slice(taps - 1 - t, taps - 1 - t + nz)
        acc[:, t : t + k] += rx.w[t] * g[rows]
    return float(np.mean(np.sum(acc**2, axis=1)) / rx.f0**2)


@dataclass
class Budget:
    keys: list
    est: np.ndarray
    est_se: np.ndarray
    truth: np.ndarray
    flagged: set
    measured_mse: float

    @property
    def total_est(self) -> float:
        return float(np.sum(self.est))

    def share(self, which="est") -> np.ndarray:
        v = self.est if which == "est" else self.truth
        return 100 * v / np.sum(v)

    def share_se(self) -> np.ndarray:
        return 100 * self.est_se / np.sum(self.est)

    def snr_db(self, which="est") -> float:
        v = self.est if which == "est" else self.truth
        return float(10 * np.log10(EA / np.sum(v)))


def residual_isi(rx: Receiver, lay, levels) -> float:
    x = evaluate(rx, lay, levels, Params.zero())
    z = slicer_error(rx, x)
    m = ffe_symbol_index(lay, z.size) - lay.m_lo
    return float(np.mean((z - levels[m]) ** 2))


def estimated_budget(mf: MeanFit, nf: NoiseFit, op: int = 0):
    f = mf.fits[op]
    rx = f.rx
    J = f.J
    nuis = J[:, NS : NS + 3]
    theta = theta_shared(f.params)
    cov_sh = mf.cov[:NS, :NS]
    est, se = {}, {}
    for g in ("dac_lane", "thd", "adc_lane", "pj"):
        if g == "pj":
            cols = [NS + 3, NS + 4]
            th = np.array([f.params.pj_c, f.params.pj_s])
            base = NS + op * len(LOCAL)
            S = mf.cov[np.ix_([base + 3, base + 4], [base + 3, base + 4])]
        else:
            cols = shared_columns(g)
            th = theta[cols]
            S = cov_sh[np.ix_(cols, cols)]
        M = _ffe_cols(rx, _project_out(J[:, cols], nuis))
        G = M.T @ M / M.shape[0]
        c = float(th @ G @ th - np.trace(G @ S))
        v = float(4 * th @ G @ S @ G @ th + 2 * np.trace(G @ S @ G @ S))
        est[g], se[g] = max(c, 0.0), np.sqrt(v)
    gain2 = float(np.mean(f.jac["gain"] ** 2))
    taps = rx.w.size
    R = rx.rx_acov[np.abs(np.subtract.outer(np.arange(taps), np.arange(taps)))]
    k_rx = gain2 * float(rx.w @ R @ rx.w) / rx.f0**2
    k_adc = float(rx.w @ rx.w) / rx.f0**2
    k_rj = rj_gain(rx, f.jac)
    kv = np.array([k_rx, k_adc, k_rj])
    sv = np.array([nf.s_rx, nf.s_adc, nf.s_rj])
    for key, i in (("rx_noise", 0), ("adc_noise", 1), ("rj", 2)):
        est[key] = max(kv[i] * sv[i], 0.0)
        se[key] = kv[i] * np.sqrt(nf.cov[i, i])
    est["quant"] = k_adc * LSB**2 / 12
    se["quant"] = 0.0
    est["isi"] = residual_isi(rx, f.lay, f.levels)
    se["isi"] = 0.0
    return est, se, dict(k_rx=k_rx, k_adc=k_adc, k_rj=k_rj)


def measured_mse(f) -> float:
    """Slicer MSE of the capture itself after the DSP's gain and threshold adaptation (LS fit of z on [a, 1])."""

    z = slicer_error(f.rx, f.x)
    m = ffe_symbol_index(f.lay, z.size) - f.lay.m_lo
    A = np.column_stack([f.levels[m], np.ones(z.size)])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    return float(np.mean((z - A @ coef) ** 2) / coef[0] ** 2)


def ablation_truth(rx: Receiver, truth: Impairments, pj_phase: float, seeds=(101, 102, 103, 104)):
    """Simulated ablation: switch on one source at a time, measure its slicer MSE."""

    lay = make_layout(rx.setting)
    sym = prbs_symbols(lay.m_hi + 8)
    levels = LEVELS[sym[lay.m_lo : lay.m_hi]]
    nom = evaluate(rx, lay, levels, Params.zero(), need_jac=True)
    nuis = np.column_stack([nom["y"], np.ones(nom["y"].size), nom["dydt"]])

    def run(include, noise, quantize, seed):
        x, _, _ = simulate(
            rx, truth, np.random.default_rng(seed), include=include, noise=noise,
            quantize=quantize, pj_phase=pj_phase, return_x=True,
        )
        return x

    clean = run(set(), (), False, 0)
    out = {}
    for g in ("dac_lane", "thd", "adc_lane", "pj"):
        d = run({g}, (), False, 0) - clean
        d = _project_out(d[:, None], nuis)[:, 0]
        out[g] = float(np.mean(slicer_error(rx, d) ** 2))
    for key, noise in (("rx_noise", ("rx",)), ("adc_noise", ("adc",)), ("rj", ("rj",))):
        out[key] = float(np.mean([np.mean(slicer_error(rx, run(set(), noise, False, s) - clean) ** 2) for s in seeds]))
    out["quant"] = float(np.mean(slicer_error(rx, run(set(), (), True, 0) - clean) ** 2))
    z = slicer_error(rx, clean)
    m = ffe_symbol_index(lay, z.size) - lay.m_lo
    out["isi"] = float(np.mean((z - levels[m]) ** 2))
    return out


@dataclass
class Identifiability:
    names: list
    eig_single: dict
    eig_joint: np.ndarray
    null_single: dict
    aliases: list
    noise_corr_single: dict
    noise_corr_joint: float
    se_single: dict


def _normalized_fisher(blocks):
    A = sum(blocks)
    d = np.sqrt(np.diag(A))
    return A / np.outer(d, d)


def identifiability(mf: MeanFit, nf: NoiseFit, null_tol: float = 1e-3) -> Identifiability:
    """Fisher information of each capture alone vs all captures jointly (shared + that capture's locals)."""

    names = [f"{n}[{k}]" for n, k in SHARED] + [n for n, _ in LOCAL]
    eig_single, null_single, se_single = {}, {}, {}
    aliases = []
    from estimator import GROUP_OF

    for c, f in enumerate(mf.fits):
        A = f.J.T @ f.J
        An = _normalized_fisher([A])
        w, V = np.linalg.eigh(An)
        eig_single[c] = w
        nulls = []
        for i in np.nonzero(w < null_tol)[0]:
            load = V[:, i] ** 2
            top = np.argsort(load)[::-1]
            grp = []
            for j in top:
                if load[j] < 0.05:
                    break
                grp.append((names[j], float(load[j])))
            nulls.append(dict(eig=float(w[i]), loadings=grp))
            groups = sorted({GROUP_OF.get(n.split("[")[0], n.split("[")[0]) for n, _ in grp})
            if c == 0:
                aliases.append(dict(eig=float(w[i]), groups=groups, loadings=grp))
        null_single[c] = nulls
        resid_var = float(np.var(f.resid))
        se_single[c] = np.sqrt(np.clip(np.diag(np.linalg.pinv(A, rcond=1e-10)), 0, None) * resid_var)
    ns, nl = NS, len(LOCAL)
    ntot = ns + len(mf.fits) * nl
    Aj = np.zeros((ntot, ntot))
    for c, f in enumerate(mf.fits):
        sel = np.r_[np.arange(ns), ns + c * nl + np.arange(nl)]
        Aj[np.ix_(sel, sel)] += f.J.T @ f.J
    eig_joint = np.linalg.eigvalsh(_normalized_fisher([Aj]))
    corr_single = {}
    for c, X in enumerate(nf.design_by_capture):
        Xs = X / np.sqrt(np.sum(X * X, axis=0))
        C = np.linalg.inv(Xs.T @ Xs)
        corr_single[c] = float(C[0, 1] / np.sqrt(C[0, 0] * C[1, 1]))
    Cj = nf.cov
    corr_joint = float(Cj[0, 1] / np.sqrt(Cj[0, 0] * Cj[1, 1]))
    return Identifiability(names, eig_single, eig_joint, null_single, aliases, corr_single, corr_joint, se_single)


def lane_report(mf: MeanFit, truth: Impairments):
    """Per-lane estimates, standard errors and truth in display units."""

    p = mf.fits[0].params
    C = mf.cov
    spec = [
        ("adc_off", 0, B8, p.adc_off / LSB, 1 / LSB, truth.adc_off_lsb, "LSB"),
        ("adc_gain", 7, B8, p.adc_gain * 100, 100, truth.adc_gain_pct, "%"),
        ("adc_skew", 14, B8, p.adc_skew * 1e3, 1e3, truth.adc_skew_mui, "mUI"),
        ("dac_off", 21, B4, p.dac_off * 100, 100, truth.dac_off_pct, "%"),
        ("dac_gain", 24, B4, p.dac_gain * 100, 100, truth.dac_gain_pct, "%"),
        ("dac_skew", 27, B4, p.dac_skew * 1e3, 1e3, truth.dac_skew_mui, "mUI"),
    ]
    out = {}
    for name, start, B, est, scale, tru, unit in spec:
        k = B.shape[1]
        cov = B @ C[start : start + k, start : start + k] @ B.T * scale**2
        out[name] = dict(est=est, cov=cov, se=np.sqrt(np.diag(cov)), truth=np.asarray(tru), unit=unit)
    return out


def _rms(est, cov):
    w = est.size
    m2 = float(np.mean(est**2) - np.trace(cov) / w)
    r = np.sqrt(max(m2, 0.0))
    g = est / (w * max(r, 1e-12))
    return r, float(np.sqrt(g @ cov @ g))


def parameter_table(mf: MeanFit, nf: NoiseFit, truth: Impairments, lanes: dict):
    rows = []
    p = mf.fits[0].params
    se = np.sqrt(np.diag(mf.cov))

    def add(section, label, est, s, tru, unit, fmt):
        rows.append(dict(section=section, label=label, est=est, se=s, truth=tru, z=(est - tru) / s, unit=unit, fmt=fmt))

    add("TX", "TX DAC 二阶 a2", p.a2, se[NS - 2], truth.a2, "", "{:.4f}")
    add("TX", "a3 三阶压缩", p.a3, se[NS - 1], truth.a3, "", "{:.4f}")
    for key, label in (("dac_gain", "DAC 各路增益误差 rms"), ("dac_skew", "边沿时序偏差 rms"), ("dac_off", "各路偏移 rms")):
        L = lanes[key]
        r, s = _rms(L["est"], L["cov"])
        add("TX", label, r, s, float(np.sqrt(np.mean(L["truth"] ** 2))), L["unit"], "{:.4f}" if L["unit"] == "%" else "{:.2f}")
    rj_ui = np.sqrt(max(nf.s_rj, 0))
    add("时钟", "TX 边沿随机抖动", rj_ui * 1e3, np.sqrt(nf.cov[2, 2]) / (2 * rj_ui) * 1e3, truth.rj_mui, "mUI rms", "{:.3f}")
    nl = len(LOCAL)
    amps, vars_ = [], []
    for c, f in enumerate(mf.fits):
        b = NS + c * nl
        S = mf.cov[np.ix_([b + 3, b + 4], [b + 3, b + 4])]
        th = np.array([f.params.pj_c, f.params.pj_s])
        a = float(np.hypot(*th))
        g = th / a
        amps.append(a)
        vars_.append(float(g @ S @ g))
    wts = 1 / np.array(vars_)
    pj = float(np.sum(wts * amps) / np.sum(wts)) * 1e3
    add("时钟", "周期抖动 PJ 幅度", pj, 1e3 / np.sqrt(np.sum(wts)), truth.pj_mui, "mUI", "{:.3f}")
    rows.append(dict(section="时钟", label="PJ 频率 (扫描峰值)", est=mf.pj_freq, se=float("nan"), truth=truth.pj_freq, z=float("nan"), unit="cyc/UI", fmt="{:.5f}"))
    rx_mv = np.sqrt(max(nf.s_rx, 0)) * 1e3
    add("RX", "RX 输入噪声", rx_mv, np.sqrt(nf.cov[0, 0]) / (2 * rx_mv / 1e3) * 1e3, truth.rx_noise_mv, "mV rms", "{:.3f}")
    adc = np.sqrt(max(nf.s_adc, 0)) / LSB
    add("ADC", "ADC 热噪声", adc, np.sqrt(nf.cov[1, 1]) / (2 * adc * LSB) / LSB, truth.adc_noise_lsb, "LSB rms", "{:.3f}")
    for key, label in (("adc_off", "ADC 各路偏移 rms"), ("adc_gain", "ADC 各路增益 rms"), ("adc_skew", "ADC 各路时序 rms")):
        L = lanes[key]
        r, s = _rms(L["est"], L["cov"])
        add("ADC", label, r, s, float(np.sqrt(np.mean(L["truth"] ** 2))), L["unit"], "{:.3f}")
    return rows
