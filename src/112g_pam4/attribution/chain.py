"""Forward model: the 112G PAM4 chain from ``model.py`` with interleaving and
analog impairments, evaluated at the actual ADC sampling instants.

Signal path for symbol index m and ADC sample index n (times in UI):

    a[m]           PRBS13Q level, LEVELS = (−1, −1/3, +1/3, +1)
    u[m]           TX FFE output, Σ c_j a[m−j]  (|u| ≤ 1)
    v[m]           DAC level: (1 + g_L(m)) · (u + a2·u² − a3·u³) + o_L(m),  L(m) = (m + dac_phase) mod Wt
    e[m]           edge time of symbol m: τ_L(m) + RJ draw
    y(t)           Σ_m v[m] · P_m(t),   P_m(t) = S(t − m − e[m]) − S(t − m − 1 − e[m+1])
    t_n            n + n_lat + t0 + φ + δ_A(n) + PJ(n),  A(n) = (n + adc_phase) mod W
    x[n]           (1 + γ)(1 + G_A(n)) · (y(t_n) + w_rx(t_n)) + O_A(n)·LSB + β + w_adc[n]
    code[n]        round(x / LSB), clipped to the ADC range

S(t) is the step response of TX driver → channel → RX front end → CTLE → VGA,
from the same stages as ``model.py``. w_rx is RX input noise (before the VGA)
shaped by the RX chain; w_adc is ADC thermal noise. γ, β, φ are per-capture
nuisances (AGC gain, DC level, CDR phase) that the receiver removes anyway.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import (  # noqa: E402
    BAUD,
    EA,
    LEVELS,
    NFPRE,
    OS,
    POST,
    PRE,
    PS,
    T0,
    TX_FFE,
    UI,
    VPK,
    analyze_link,
    cascade,
    channel_stages,
    design,
    rx_stages,
)

W_ADC = 8
W_DAC = 4
ADC_BITS = 8
LSB = 2.0 / 2**ADC_BITS
CODE_MIN = -(2 ** (ADC_BITS - 1))
CODE_MAX = 2 ** (ADC_BITS - 1) - 1

OSF = 256
NWIN = 256
NFFT_F = OSF * NWIN
# Lags k = n + n_lat − m kept per sample: one UI before the edge to the end of the pulse tail.
KMIN, KMAX = -8, 58
KS = np.arange(KMIN, KMAX + 1)
PRBS_SEED = 0x1D3


def _impulse(stages, gain: float, osf: int, nfft: int) -> np.ndarray:
    """h(t)·dt on a grid of `osf` samples per UI, delayed by T0 UI; sum = H(0)·gain."""

    k = np.arange(nfft // 2 + 1)
    f = k * BAUD * osf / nfft
    lm, ph = cascade(stages, f)
    ph = ph - 2 * np.pi * f * T0 * UI
    return np.fft.irfft(np.exp(lm + 1j * ph) * gain, n=nfft)


class StepTable:
    """Step response S(t) and its slope S'(t) (per UI) on a 1/256 UI grid."""

    def __init__(self, stages, gain: float):
        h = _impulse(stages, gain, OSF, NFFT_F)
        self.s = np.cumsum(h)
        self.ds = h * OSF
        self.grid = np.arange(self.s.size, dtype=np.float64)

    def _index(self, t):
        return (np.asarray(t, dtype=np.float64) + T0) * OSF

    def step(self, t):
        return np.interp(self._index(t), self.grid, self.s)

    def slope(self, t):
        return np.interp(self._index(t), self.grid, self.ds)

    def pulse(self, t):
        return self.step(t) - self.step(t - 1.0)


@dataclass
class Impairments:
    """True (simulated) error sources. Lane vectors are zero-mean mismatches."""

    adc_off_lsb: np.ndarray
    adc_gain_pct: np.ndarray
    adc_skew_mui: np.ndarray
    dac_off_pct: np.ndarray
    dac_gain_pct: np.ndarray
    dac_skew_mui: np.ndarray
    a2: float = 0.02
    a3: float = 0.05
    rj_mui: float = 8.0
    # Not shown in the screenshot; chosen so RX noise ≈ 62 % and PJ ≈ 1.6 % of the budget as displayed there.
    pj_mui: float = 11.0
    pj_freq: float = 0.0371
    rx_noise_mv: float = 0.95
    adc_noise_lsb: float = 0.42

    def demeaned(self) -> "Impairments":
        z = lambda v: np.asarray(v, dtype=np.float64) - np.mean(v)  # noqa: E731
        return replace(
            self,
            adc_off_lsb=z(self.adc_off_lsb),
            adc_gain_pct=z(self.adc_gain_pct),
            adc_skew_mui=z(self.adc_skew_mui),
            dac_off_pct=z(self.dac_off_pct),
            dac_gain_pct=z(self.dac_gain_pct),
            dac_skew_mui=z(self.dac_skew_mui),
        )


def default_truth() -> Impairments:
    """Lane values read off the black truth ticks in the screenshot; scalars likewise."""

    return Impairments(
        adc_off_lsb=np.array([-0.12, -0.70, -0.40, -0.29, 0.52, 1.12, -0.07, -0.06]),
        adc_gain_pct=np.array([0.13, 1.15, 0.02, -0.12, -0.50, 0.05, -0.45, -0.27]),
        adc_skew_mui=np.array([-13.3, 7.0, 17.3, 2.0, -4.3, 4.0, -11.5, -1.2]),
        dac_off_pct=np.array([0.15, 1.12, -0.75, -0.52]),
        dac_gain_pct=np.array([-0.20, -0.75, 1.27, -0.32]),
        dac_skew_mui=np.array([-1.6, -21.0, 14.2, 8.4]),
    ).demeaned()


@dataclass
class CaptureSetting:
    """What the analyzer is told about one capture (the receiver is known)."""

    name: str
    gdc: float
    gdc2: float
    tx_ffe: bool
    adc_phase: int
    dac_phase: int
    m_start: int
    seed: int
    n_samples: int = 16384
    loss_db: float = 28.0


def default_captures(n_samples: int = 16384) -> list[CaptureSetting]:
    """Four captures. Each changes one known thing so a different alias breaks.

    rotation = (m_start + dac_phase − adc_phase) mod 4 is the DAC-to-ADC lane alignment.
    """

    return [
        CaptureSetting("A 基准", -17, -3, True, 0, 0, 1200, 11, n_samples),
        CaptureSetting("B 交织相位 +1", -17, -3, True, 0, 1, 1200, 12, n_samples),
        CaptureSetting("C CTLE −12 dB", -12, -3, True, 0, 2, 1200, 13, n_samples),
        CaptureSetting("D TX FFE 关", -17, -3, False, 0, 3, 1200, 14, n_samples),
    ]


@dataclass
class Receiver:
    """Known channel + ADC receiver for one setting."""

    setting: CaptureSetting
    vga: float
    t0: float
    table: StepTable
    rx_fir32: np.ndarray
    h: np.ndarray
    w: np.ndarray
    f0: float
    ffe_c: np.ndarray
    rx_acov: np.ndarray = field(default=None)


def build_receiver(s: CaptureSetting) -> Receiver:
    a = analyze_link(
        loss_db=s.loss_db,
        tx_ffe=s.tx_ffe,
        auto_ctle=False,
        gdc=s.gdc,
        gdc2=s.gdc2,
        dsp=True,
        dfe=False,
        shared_front_end=False,
    )
    stages = channel_stages(s.loss_db) + rx_stages(s.gdc, s.gdc2)
    table = StepTable(stages, a.vga)
    # model.pulse's discrete rectangle is centred (OS−1)/2 samples in, 1/(2·OS) UI ahead of a continuous [0, 1) UI box.
    t0 = (PS + a.ts_off - T0 * OS) / OS + 0.5 / OS
    ffe_c = TX_FFE if s.tx_ffe else np.array([0.0, 1.0, 0.0])
    # Baud cursors of the full TX-symbol response (TX FFE included), at this sampling phase.
    k = np.arange(-PRE, POST + 1)
    p = table.pulse(t0 + k + 1) * ffe_c[0] + table.pulse(t0 + k) * ffe_c[1] + table.pulse(t0 + k - 1) * ffe_c[2]
    w, _ = design(p, a.noise, True, False)
    f = np.convolve(w, p)
    f0 = float(f[PRE + NFPRE])
    # RX input noise path: white at the pad, through RX pole + CTLE + VGA, in FS per volt.
    rx_fir32 = _impulse(rx_stages(s.gdc, s.gdc2), a.vga / VPK, OS, 8192)
    rx_fir32 = rx_fir32[: 64 * OS]
    lags = np.arange(0, 16)
    var_fine = 32.0  # per-volt² variance of one OS=32 fine sample (flat to 16·BAUD)
    acov = np.array([var_fine * np.dot(rx_fir32[: rx_fir32.size - l * OS], rx_fir32[l * OS :]) for l in lags])
    return Receiver(s, a.vga, t0, table, rx_fir32, p, w, f0, ffe_c, acov)


class Prbs13Q:
    """PRBS13 x¹³+x¹²+x²+x+1, bit pairs Gray-mapped (00→0, 01→1, 11→2, 10→3)."""

    def __init__(self, seed: int = PRBS_SEED):
        self.s = seed & 0x1FFF or 1

    def bit(self) -> int:
        s = self.s
        b = ((s >> 12) ^ (s >> 11) ^ (s >> 1) ^ s) & 1
        self.s = ((s << 1) | b) & 0x1FFF
        return b

    def symbols(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        for i in range(n):
            msb, lsb = self.bit(), self.bit()
            out[i] = (2 if lsb else 3) if msb else (1 if lsb else 0)
        return out


_PRBS_CACHE: dict[int, np.ndarray] = {}


def prbs_symbols(n: int, seed: int = PRBS_SEED) -> np.ndarray:
    key = seed
    cached = _PRBS_CACHE.get(key)
    if cached is None or cached.size < n:
        cached = Prbs13Q(seed).symbols(max(n, 8191 * 4))
        _PRBS_CACHE[key] = cached
    return cached[:n]


@dataclass
class Params:
    """Physical parameter set used by both the simulator and the analyzer's model.

    Units: offsets in FS (ADC) or level units (DAC); gains as fractions; times in UI.
    """

    adc_off: np.ndarray
    adc_gain: np.ndarray
    adc_skew: np.ndarray
    dac_off: np.ndarray
    dac_gain: np.ndarray
    dac_skew: np.ndarray
    a2: float = 0.0
    a3: float = 0.0
    pj_c: float = 0.0
    pj_s: float = 0.0
    pj_freq: float = 0.0
    gamma: float = 0.0
    beta: float = 0.0
    phi: float = 0.0

    @staticmethod
    def zero() -> "Params":
        return Params(
            np.zeros(W_ADC), np.zeros(W_ADC), np.zeros(W_ADC), np.zeros(W_DAC), np.zeros(W_DAC), np.zeros(W_DAC)
        )


def params_from_truth(t: Impairments, pj_phase: float, include=None) -> Params:
    """Map an Impairments truth to model Params. `include` limits to named groups (ablation)."""

    groups = {"adc_lane", "dac_lane", "thd", "pj"} if include is None else include
    p = Params.zero()
    if "adc_lane" in groups:
        p.adc_off = t.adc_off_lsb * LSB
        p.adc_gain = t.adc_gain_pct / 100
        p.adc_skew = t.adc_skew_mui / 1000
    if "dac_lane" in groups:
        p.dac_off = t.dac_off_pct / 100
        p.dac_gain = t.dac_gain_pct / 100
        p.dac_skew = t.dac_skew_mui / 1000
    if "thd" in groups:
        p.a2 = t.a2
        p.a3 = t.a3
    if "pj" in groups:
        amp = t.pj_mui / 1000
        p.pj_c = amp * np.cos(pj_phase)
        p.pj_s = amp * np.sin(pj_phase)
        p.pj_freq = t.pj_freq
    return p


@dataclass
class Layout:
    """Index bookkeeping for one capture: which symbols feed which samples."""

    n: np.ndarray
    m_of_n: np.ndarray
    M: np.ndarray
    adc_lane: np.ndarray
    dac_lane_M: np.ndarray
    m_lo: int
    m_hi: int


def make_layout(s: CaptureSetting) -> Layout:
    n = np.arange(s.n_samples)
    m_of_n = n + s.m_start
    M = m_of_n[:, None] - KS[None, :]
    return Layout(
        n=n,
        m_of_n=m_of_n,
        M=M,
        adc_lane=(n + s.adc_phase) % W_ADC,
        dac_lane_M=(M + s.dac_phase) % W_DAC,
        m_lo=int(M.min()),
        m_hi=int(M.max()) + 2,
    )


def dac_levels(u: np.ndarray, m_idx: np.ndarray, s: CaptureSetting, p: Params):
    """DAC output v[m] and its pieces, for symbol indices m_idx."""

    lane = (m_idx + s.dac_phase) % W_DAC
    nl = u + p.a2 * u * u - p.a3 * u**3
    v = (1 + p.dac_gain[lane]) * nl + p.dac_off[lane]
    return v, nl, lane


def tx_ffe_output(a: np.ndarray, c: np.ndarray) -> np.ndarray:
    """u[m] = c(−1)·a[m+1] + c(0)·a[m] + c(+1)·a[m−1] (array index = m − m_lo)."""

    u = c[1] * a
    u[:-1] += c[0] * a[1:]
    u[1:] += c[2] * a[:-1]
    return u


def sample_times(rx: Receiver, lay: Layout, p: Params) -> np.ndarray:
    pj = p.pj_c * np.sin(2 * np.pi * p.pj_freq * lay.n) + p.pj_s * np.cos(2 * np.pi * p.pj_freq * lay.n)
    return lay.m_of_n + rx.t0 + p.phi + p.adc_skew[lay.adc_lane] + pj


def evaluate(rx: Receiver, lay: Layout, levels: np.ndarray, p: Params, edge_noise=None, need_jac=False):
    """Deterministic ADC input x[n] for parameters p (optionally with per-edge RJ draws).

    `levels` are a[m] for m in [m_lo, m_hi) as PAM4 level values.
    Returns x and, if need_jac, the pieces the analyzer needs for derivatives.
    """

    s = rx.setting
    m_all = np.arange(lay.m_lo, lay.m_hi)
    u = tx_ffe_output(levels.astype(np.float64).copy(), rx.ffe_c)
    v, nl, lane_all = dac_levels(u, m_all, s, p)
    e = p.dac_skew[lane_all].copy()
    if edge_noise is not None:
        e = e + edge_noise
    t_n = sample_times(rx, lay, p)
    idx = lay.M - lay.m_lo
    arg1 = t_n[:, None] - lay.M - e[idx]
    arg2 = t_n[:, None] - lay.M - 1.0 - e[idx + 1]
    s1 = rx.table.step(arg1)
    s2 = rx.table.step(arg2)
    pulse = s1 - s2
    y = np.einsum("nk,nk->n", v[idx], pulse)
    gain = (1 + p.gamma) * (1 + p.adc_gain[lay.adc_lane])
    x = gain * y + p.adc_off[lay.adc_lane] + p.beta
    if not need_jac:
        return x
    g1 = rx.table.slope(arg1)
    g2 = rx.table.slope(arg2)
    vprev = np.concatenate(([v[0]], v[:-1]))
    d = v - vprev
    return dict(
        x=x,
        y=y,
        gain=gain,
        pulse=pulse,
        g1=g1,
        dydt=np.einsum("nk,nk->n", v[idx], g1 - g2),
        idx=idx,
        v=v,
        u=u,
        nl=nl,
        d=d,
        lane_all=lane_all,
        t_n=t_n,
    )


def rx_noise_samples(rx: Receiver, lay: Layout, sigma_v: float, rng: np.random.Generator, t_n: np.ndarray) -> np.ndarray:
    """RX input noise (white at the pad, flat to 16·BAUD) through RX + CTLE + VGA, sampled at t_n."""

    n_fine = (lay.n.size + 80) * OS
    white = rng.standard_normal(n_fine) * sigma_v * np.sqrt(32.0)
    nf = 1 << int(np.ceil(np.log2(n_fine + rx.rx_fir32.size)))
    colored = np.fft.irfft(np.fft.rfft(white, nf) * np.fft.rfft(rx.rx_fir32, nf), nf)[:n_fine]
    t_rel = t_n - lay.m_of_n[0] + 40
    pos = np.clip(np.round(t_rel * OS).astype(np.int64), 0, n_fine - 1)
    return colored[pos]


@dataclass
class Capture:
    setting: CaptureSetting
    codes: np.ndarray
    # Simulator-only truth, never read by the analyzer.
    truth_levels: np.ndarray
    truth_pj_phase: float
    clipped: int


def simulate(
    rx: Receiver,
    truth: Impairments,
    rng: np.random.Generator,
    include=None,
    noise=("rx", "adc", "rj"),
    quantize=True,
    pj_phase=None,
    return_x=False,
):
    """Generate one capture of ADC codes. `include` / `noise` select sources (ablation)."""

    s = rx.setting
    lay = make_layout(s)
    sym = prbs_symbols(lay.m_hi + 8)
    levels = LEVELS[sym[lay.m_lo : lay.m_hi]]
    if pj_phase is None:
        pj_phase = float(rng.uniform(0, 2 * np.pi))
    p = params_from_truth(truth, pj_phase, include)
    edge = None
    if "rj" in noise:
        edge = rng.standard_normal(lay.m_hi - lay.m_lo) * truth.rj_mui / 1000
    x = evaluate(rx, lay, levels, p, edge_noise=edge)
    if "rx" in noise:
        t_n = sample_times(rx, lay, p)
        gain = (1 + p.gamma) * (1 + p.adc_gain[lay.adc_lane])
        x = x + gain * rx_noise_samples(rx, lay, truth.rx_noise_mv * 1e-3, rng, t_n)
    if "adc" in noise:
        x = x + rng.standard_normal(x.size) * truth.adc_noise_lsb * LSB
    if return_x and not quantize:
        return x, levels, p
    raw = np.round(x / LSB)
    clipped = int(np.sum((raw < CODE_MIN) | (raw > CODE_MAX)))
    codes = np.clip(raw, CODE_MIN, CODE_MAX).astype(np.int16)
    if return_x:
        return codes.astype(np.float64) * LSB, levels, p
    return Capture(s, codes, levels, pj_phase, clipped)


def slicer_error(rx: Receiver, err: np.ndarray) -> np.ndarray:
    """FFE-referred error: the receiver's 12-tap FFE applied to an ADC-domain error, ÷ f0."""

    return np.convolve(err, rx.w, mode="valid") / rx.f0


def ffe_output(rx: Receiver, x: np.ndarray) -> np.ndarray:
    """z[j] for symbol m_of_n[j + NF − 1 − NFPRE], matching slicer_error's 'valid' indexing."""

    return np.convolve(x, rx.w, mode="valid") / rx.f0


def ffe_symbol_index(lay: Layout, nz: int) -> np.ndarray:
    """Symbol index decided by FFE output j (valid convolution, 12 taps, main tap NFPRE)."""

    j = np.arange(nz)
    return lay.m_of_n[j + 11 - NFPRE]


__all__ = [
    "EA",
    "LEVELS",
    "LSB",
    "W_ADC",
    "W_DAC",
    "Impairments",
    "CaptureSetting",
    "Receiver",
    "Params",
    "Layout",
    "build_receiver",
    "default_captures",
    "default_truth",
    "evaluate",
    "simulate",
    "make_layout",
    "prbs_symbols",
    "slicer_error",
    "ffe_output",
    "ffe_symbol_index",
]
