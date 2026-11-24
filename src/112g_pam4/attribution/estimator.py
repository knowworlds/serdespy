"""Analyzer: infer error sources from ADC codes plus the known channel and ADC receiver.

Steps
1. Sync: equalize with the known FFE, slice, Gray-demap, check the PRBS13 recurrence,
   locate the PRBS phase. After this the transmitted symbols are known.
2. Mean model: joint Gauss–Newton least squares over all captures for the
   deterministic impairments (lane mismatch, DAC THD, PJ) plus per-capture nuisances.
3. Noise model: method of moments on residual auto-products for RX input noise,
   ADC thermal noise and TX edge RJ.
4. Covariance: block (HAC) sandwich, so standard errors hold with colored residuals.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from chain import (
    LEVELS,
    LSB,
    W_ADC,
    W_DAC,
    Layout,
    Params,
    Receiver,
    evaluate,
    ffe_output,
    make_layout,
    prbs_symbols,
)

BLOCK = 256
GRAY_BITS = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
NATURAL_BITS = {0: (0, 0), 1: (0, 1), 2: (1, 0), 3: (1, 1)}
PRBS_PERIOD = 8191


def contrast_basis(w: int) -> np.ndarray:
    """Orthonormal W×(W−1) basis of zero-mean lane vectors (Helmert)."""

    b = np.zeros((w, w - 1))
    for j in range(1, w):
        b[:j, j - 1] = 1.0
        b[j, j - 1] = -j
        b[:, j - 1] /= np.linalg.norm(b[:, j - 1])
    return b


B8 = contrast_basis(W_ADC)
B4 = contrast_basis(W_DAC)

SHARED = (
    [("adc_off", i) for i in range(W_ADC - 1)]
    + [("adc_gain", i) for i in range(W_ADC - 1)]
    + [("adc_skew", i) for i in range(W_ADC - 1)]
    + [("dac_off", i) for i in range(W_DAC - 1)]
    + [("dac_gain", i) for i in range(W_DAC - 1)]
    + [("dac_skew", i) for i in range(W_DAC - 1)]
    + [("a2", 0), ("a3", 0)]
)
LOCAL = [("gamma", 0), ("beta", 0), ("phi", 0), ("pj_c", 0), ("pj_s", 0)]
NUISANCE = ("gamma", "beta", "phi")
GROUP_OF = {
    "adc_off": "adc_lane",
    "adc_gain": "adc_lane",
    "adc_skew": "adc_lane",
    "dac_off": "dac_lane",
    "dac_gain": "dac_lane",
    "dac_skew": "dac_lane",
    "a2": "thd",
    "a3": "thd",
    "pj_c": "pj",
    "pj_s": "pj",
}


@dataclass
class SyncResult:
    mapping: str
    recurrence_pass: dict
    m_start: int
    symbol_errors: int
    symbols_checked: int
    linear_r2: float = float("nan")


def _bits(decisions: np.ndarray, table: dict) -> np.ndarray:
    lut = np.array([table[i] for i in range(4)], dtype=np.int8)
    return lut[decisions].reshape(-1)


def _recurrence_ok(bits: np.ndarray) -> np.ndarray:
    """PRBS13 x¹³+x¹²+x²+x+1: b[i] = b[i−13] ⊕ b[i−12] ⊕ b[i−2] ⊕ b[i−1]."""

    return bits[13:] == (bits[:-13] ^ bits[1:-12] ^ bits[11:-2] ^ bits[12:-1])


def slice_levels(z: np.ndarray) -> np.ndarray:
    return np.digitize(z, [-2 / 3, 0.0, 2 / 3])


def sync(rx: Receiver, codes: np.ndarray) -> SyncResult:
    """Find the bit mapping and PRBS phase from ADC codes alone."""

    x = codes.astype(np.float64) * LSB
    z = ffe_output(rx, x)
    dec = slice_levels(z)
    rates = {}
    for name, table in (("gray", GRAY_BITS), ("natural", NATURAL_BITS)):
        rates[name] = float(np.mean(_recurrence_ok(_bits(dec, table))))
    mapping = max(rates, key=rates.get)
    ok = _recurrence_ok(_bits(dec, GRAY_BITS if mapping == "gray" else NATURAL_BITS))
    # First 48-symbol window whose recurrence holds everywhere.
    win = 48
    sym_ok = np.ones(dec.size, dtype=bool)
    bad = np.nonzero(~ok)[0] + 13
    sym_ok[np.unique(bad // 2)] = False
    j0 = None
    run = 0
    for j in range(dec.size):
        run = run + 1 if sym_ok[j] else 0
        if run >= win and j >= win:
            j0 = j - win + 1
            break
    if j0 is None:
        raise RuntimeError("PRBS sync failed: no clean window")
    ref = prbs_symbols(PRBS_PERIOD + win)
    view = np.lib.stride_tricks.sliding_window_view(ref, win)[:PRBS_PERIOD]
    hits = np.nonzero(np.all(view == dec[j0 : j0 + win], axis=1))[0]
    if hits.size != 1:
        raise RuntimeError(f"PRBS sync ambiguous: {hits.size} matches")
    # FFE output j decides symbol m_start + j + 8 (12 taps, main tap 3).
    m_start = int(hits[0] - j0 - 8) % PRBS_PERIOD
    known = prbs_symbols(m_start + 8 + dec.size)[m_start + 8 : m_start + 8 + dec.size]
    return SyncResult(mapping, rates, m_start, int(np.sum(known != dec)), int(dec.size))


@dataclass
class CaptureFit:
    rx: Receiver
    lay: Layout
    levels: np.ndarray
    x: np.ndarray
    params: Params = None
    J: np.ndarray = None
    resid: np.ndarray = None
    jac: dict = None
    use_pj: bool = False


def _lane_cols(mask_lane: np.ndarray, values: np.ndarray, w: int) -> np.ndarray:
    """N×W columns: Σ over k of values[n,k] where lane(M[n,k]) == j (or per-sample lanes)."""

    if values.ndim == 1:
        out = np.zeros((values.size, w))
        out[np.arange(values.size), mask_lane] = values
        return out
    out = np.empty((values.shape[0], w))
    for j in range(w):
        out[:, j] = np.sum(np.where(mask_lane == j, values, 0.0), axis=1)
    return out


def jacobian(fit: CaptureFit, pj_freq: float):
    """Analytic ∂x/∂θ at fit.params, columns ordered SHARED then LOCAL."""

    p = fit.params
    lay = fit.lay
    j = evaluate(fit.rx, lay, fit.levels, p, need_jac=True)
    idx = j["idx"]
    gain = j["gain"]
    g_lane = 1 + p.dac_gain[j["lane_all"]]
    dlane = lay.dac_lane_M
    pulse = j["pulse"]
    gp = gain[:, None]
    n = lay.n
    cols = []
    cols.append(_lane_cols(lay.adc_lane, np.ones(n.size), W_ADC) @ B8)
    cols.append(_lane_cols(lay.adc_lane, (1 + p.gamma) * j["y"], W_ADC) @ B8)
    cols.append(_lane_cols(lay.adc_lane, gain * j["dydt"], W_ADC) @ B8)
    cols.append(_lane_cols(dlane, gp * pulse, W_DAC) @ B4)
    cols.append(_lane_cols(dlane, gp * pulse * j["nl"][idx], W_DAC) @ B4)
    cols.append(_lane_cols(dlane, -gp * j["d"][idx] * j["g1"], W_DAC) @ B4)
    u = j["u"]
    cols.append(np.sum(gp * pulse * (g_lane * u * u)[idx], axis=1)[:, None])
    cols.append(np.sum(gp * pulse * (-g_lane * u**3)[idx], axis=1)[:, None])
    dtdt = gain * j["dydt"]
    cols.append(((1 + p.adc_gain[lay.adc_lane]) * j["y"])[:, None])
    cols.append(np.ones((n.size, 1)))
    cols.append(dtdt[:, None])
    cols.append((dtdt * np.sin(2 * np.pi * pj_freq * n))[:, None])
    cols.append((dtdt * np.cos(2 * np.pi * pj_freq * n))[:, None])
    J = np.hstack(cols)
    return J, j


def _apply(fit: CaptureFit, shared: np.ndarray, local: np.ndarray, pj_freq: float) -> None:
    p = fit.params
    i = 0

    def take(k):
        nonlocal i
        v = shared[i : i + k]
        i += k
        return v

    p.adc_off = p.adc_off + B8 @ take(W_ADC - 1)
    p.adc_gain = p.adc_gain + B8 @ take(W_ADC - 1)
    p.adc_skew = p.adc_skew + B8 @ take(W_ADC - 1)
    p.dac_off = p.dac_off + B4 @ take(W_DAC - 1)
    p.dac_gain = p.dac_gain + B4 @ take(W_DAC - 1)
    p.dac_skew = p.dac_skew + B4 @ take(W_DAC - 1)
    p.a2 += take(1)[0]
    p.a3 += take(1)[0]
    p.gamma += local[0]
    p.beta += local[1]
    p.phi += local[2]
    if fit.use_pj:
        p.pj_c += local[3]
        p.pj_s += local[4]
    p.pj_freq = pj_freq


@dataclass
class MeanFit:
    fits: list
    theta: np.ndarray
    cov: np.ndarray
    names: list
    pj_freq: float
    iterations: int
    fisher: np.ndarray
    col_scale: np.ndarray


def _global_index(nc: int, use_pj: bool):
    ns = len(SHARED)
    nl = len(LOCAL)
    names = list(SHARED)
    for c in range(nc):
        names += [(f"{n}@{c}", k) for n, k in LOCAL]
    keep = np.ones(ns + nc * nl, dtype=bool)
    if not use_pj:
        for c in range(nc):
            keep[ns + c * nl + 3 : ns + c * nl + 5] = False
    return names, keep


def _assemble(fits, use_pj):
    nc = len(fits)
    ns, nl = len(SHARED), len(LOCAL)
    ntot = ns + nc * nl
    A = np.zeros((ntot, ntot))
    b = np.zeros(ntot)
    for c, f in enumerate(fits):
        sel = np.r_[np.arange(ns), ns + c * nl + np.arange(nl)]
        A[np.ix_(sel, sel)] += f.J.T @ f.J
        b[sel] += f.J.T @ f.resid
    _, keep = _global_index(nc, use_pj)
    return A, b, keep


def _scores(fits, keep):
    nc = len(fits)
    ns, nl = len(SHARED), len(LOCAL)
    ntot = ns + nc * nl
    out = []
    for c, f in enumerate(fits):
        sel = np.r_[np.arange(ns), ns + c * nl + np.arange(nl)]
        for start in range(0, f.resid.size, BLOCK):
            s = np.zeros(ntot)
            s[sel] = f.J[start : start + BLOCK].T @ f.resid[start : start + BLOCK]
            out.append(s[keep])
    return np.array(out)


def find_pj_frequency(fits, exclude=0.004) -> float:
    """Demodulate residual × slope; the PJ tone is the peak away from lane harmonics k/8."""

    power = None
    nfft = 1 << 17
    for f in fits:
        q = f.resid * f.jac["gain"] * f.jac["dydt"]
        spec = np.abs(np.fft.rfft(q - q.mean(), nfft)) ** 2
        power = spec if power is None else power + spec
    freqs = np.fft.rfftfreq(nfft)
    ok = (freqs > 0.002) & (freqs < 0.498)
    for k in range(1, 5):
        ok &= np.abs(freqs - k / 8) > exclude
    i = int(np.argmax(np.where(ok, power, 0)))
    a, b0, c = np.log(power[i - 1 : i + 2] + 1e-300)
    delta = 0.5 * (a - c) / (a - 2 * b0 + c)
    return float(freqs[i] + delta * (freqs[1] - freqs[0]))


def fit_mean(fits: list, iterations: int = 6) -> MeanFit:
    for f in fits:
        f.params = Params.zero()
        f.use_pj = False
    pj_freq = 0.0
    for it in range(iterations):
        use_pj = it >= 2
        for f in fits:
            f.use_pj = use_pj
            f.J, f.jac = jacobian(f, pj_freq)
            f.resid = f.x - f.jac["x"]
        if it == 2:
            pj_freq = find_pj_frequency(fits)
            for f in fits:
                f.J, f.jac = jacobian(f, pj_freq)
        A, b, keep = _assemble(fits, use_pj)
        Ak = A[np.ix_(keep, keep)]
        scale = 1 / np.sqrt(np.diag(Ak))
        step_k = scale * np.linalg.solve(Ak * scale[:, None] * scale[None, :], b[keep] * scale)
        step = np.zeros(A.shape[0])
        step[keep] = step_k
        ns, nl = len(SHARED), len(LOCAL)
        for c, f in enumerate(fits):
            _apply(f, step[:ns], step[ns + c * nl : ns + (c + 1) * nl], pj_freq)
    for f in fits:
        f.J, f.jac = jacobian(f, pj_freq)
        f.resid = f.x - f.jac["x"]
    A, b, keep = _assemble(fits, True)
    Ak = A[np.ix_(keep, keep)]
    scale = 1 / np.sqrt(np.diag(Ak))
    Ainv = scale[:, None] * np.linalg.inv(Ak * scale[:, None] * scale[None, :]) * scale[None, :]
    S = _scores(fits, keep)
    cov = Ainv @ (S.T @ S) @ Ainv
    names, _ = _global_index(len(fits), True)
    theta = np.zeros(int(keep.sum()))
    return MeanFit(fits, theta, cov, [n for n, k in zip(names, keep) if k], pj_freq, iterations, Ak, scale)


def shared_cov(mf: MeanFit) -> np.ndarray:
    return mf.cov[: len(SHARED), : len(SHARED)]


@dataclass
class NoiseFit:
    s_rx: float
    s_adc: float
    s_rj: float
    cov: np.ndarray
    design_by_capture: list = field(default_factory=list)


def _rj_feature(jac: dict, lag: int) -> np.ndarray:
    """z_ℓ[n] = Σ_m d_m² S'(t_n − m) S'(t_{n+ℓ} − m): RJ covariance at lag ℓ per unit σ²."""

    d2 = jac["d"][jac["idx"]] ** 2
    g = jac["g1"]
    n, k = g.shape
    if lag == 0:
        z = np.sum(d2 * g * g, axis=1)
    else:
        z = np.zeros(n)
        z[:-lag] = np.sum(d2[:-lag, : k - lag] * g[:-lag, : k - lag] * g[lag:, lag:], axis=1)
    return z


def fit_noise(fits: list, max_lag: int = 3) -> NoiseFit:
    """Moments E[r_n r_{n+ℓ}] = σ_rx²·R_rx[ℓ] + (σ_adc² + LSB²/12)·δ_ℓ + σ_rj²·z_ℓ[n]."""

    q = LSB**2 / 12
    rows_y, rows_x, block_id = [], [], []
    designs = []
    base = 0
    for f in fits:
        r = f.resid
        gain = f.jac["gain"]
        n = r.size
        blk = np.arange(n) // BLOCK + base
        xs, ys = [], []
        for lag in range(max_lag + 1):
            prod = np.zeros(n)
            prod[: n - lag] = r[: n - lag] * r[lag:]
            gg = np.zeros(n)
            gg[: n - lag] = gain[: n - lag] * gain[lag:]
            feat = np.column_stack(
                [
                    gg * f.rx.rx_acov[lag],
                    np.full(n, 1.0 if lag == 0 else 0.0),
                    gg * _rj_feature(f.jac, lag),
                ]
            )
            valid = slice(0, n - lag)
            target = prod.copy()
            if lag == 0:
                target -= q
            xs.append(feat[valid])
            ys.append(target[valid])
            rows_y.append(target[valid])
            rows_x.append(feat[valid])
            block_id.append(blk[valid])
        designs.append(np.vstack(xs))
        base = blk.max() + 1
    X = np.vstack(rows_x)
    y = np.concatenate(rows_y)
    blocks = np.concatenate(block_id)
    scale = 1 / np.sqrt(np.sum(X * X, axis=0))
    Xs = X * scale
    A = Xs.T @ Xs
    theta_s = np.linalg.solve(A, Xs.T @ y)
    res = y - Xs @ theta_s
    nb = blocks.max() + 1
    S = np.zeros((nb, 3))
    for i in range(3):
        S[:, i] = np.bincount(blocks, weights=Xs[:, i] * res, minlength=nb)
    Ainv = np.linalg.inv(A)
    cov_s = Ainv @ (S.T @ S) @ Ainv
    theta = theta_s * scale
    cov = cov_s * scale[:, None] * scale[None, :]
    return NoiseFit(float(theta[0]), float(theta[1]), float(theta[2]), cov, designs)


def prepare(rx: Receiver, codes: np.ndarray) -> tuple[CaptureFit, SyncResult]:
    s = sync(rx, codes)
    setting = rx.setting
    if s.m_start != setting.m_start:
        raise RuntimeError(f"sync found m_start={s.m_start}, expected {setting.m_start}")
    lay = make_layout(setting)
    sym = prbs_symbols(lay.m_hi + 8)
    levels = LEVELS[sym[lay.m_lo : lay.m_hi]]
    fit = CaptureFit(rx, lay, levels, codes.astype(np.float64) * LSB)
    clean = evaluate(rx, lay, levels, Params.zero())
    resid = fit.x - clean
    s.linear_r2 = float(1 - np.var(resid) / np.var(fit.x))
    return fit, s
