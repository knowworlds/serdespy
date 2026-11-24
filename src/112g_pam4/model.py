"""112G PAM4 link model recovered from the classroom animation.

The page at https://circuits-and-systems.tokenzhang.com/serdes/112g-pam4-link/
hydrates `_astro/SerdesLink.-_HQoR3o.js`, which imports `_astro/streams.Dv39vdu_.js`.
That bundle is the compiled form of the classroom sources `model.ts`, `streams.ts`
and `eyes.ts` (no source map is published). Names and equations below follow that
bundle. The browser checks its results against a NumPy `irfft` + `linalg.solve`
reference; this module uses the same two operations.

Blocks the page actually evaluates, in order:

    PRBS13Q symbols → optional 3-tap TX FFE → TX driver (2 poles at 50 GHz)
    → channel (skin, dielectric, fixed echo, optional 1 UI echo)
    → RX front end (1 pole at 45 GHz) → CTLE → VGA
    → 12-tap MMSE FFE, optional 1-tap DFE

There is no separate package S-parameter block, no bathtub, and no time-interleaved
ADC mismatch model. Random jitter enters as an equivalent noise variance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BAUD = 56e9
FN = BAUD / 2
UI = 1 / BAUD
OS = 32
NFFT = 8192
T0 = 8
PS = (T0 - 3) * OS
PLU = 56
PLEN = PLU * OS
LEVELS = np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], dtype=np.float64)
EA = 5.0 / 9.0
PRE, POST = 6, 44
NFPRE, NFPOST = 3, 8
NF = NFPRE + NFPOST + 1
TX_FFE = np.array([-0.10, 0.75, -0.15], dtype=np.float64)
VPK = 0.5
SIG_ADC = 0.01
RJ = 0.2e-12
SNR_TX_DB = 28.0
STX2 = EA * 10 ** (-SNR_TX_DB / 10)
# The bundle uses this truncated 20/ln(10), not the full expansion.
NEPER_PER_DB = 1 / 8.685889638
DB_PER_NEPER = 8.685889638
GAM = 0.9
CG = float(np.cos(GAM * np.pi / 2))
SG = float(np.sin(GAM * np.pi / 2))

KS = np.arange(-PRE - NFPRE, POST + NFPOST + 1)


def _as_f(f) -> np.ndarray:
    return np.asarray(f, dtype=np.float64)


def poles(fp: float, n: int):
    """n identical real poles: (1 + j f/fp)^(-n)."""

    def h(f):
        r = _as_f(f) / fp
        return -0.5 * n * np.log1p(r * r), -n * np.arctan(r)

    return h


def skin(loss_db: float):
    """Skin effect exp(-a √(j f/f_N)), with a chosen so the loss at f_N is loss_db."""

    a = loss_db * NEPER_PER_DB * np.sqrt(2.0)

    def h(f):
        v = -a * np.sqrt(_as_f(f) / FN) / np.sqrt(2.0)
        return v, v

    return h


def diel(loss_db: float):
    """Dielectric exp(-b (j f/f_N)^0.9). Causal, loss nearly proportional to f."""

    b = loss_db * NEPER_PER_DB / CG

    def h(f):
        v = b * (_as_f(f) / FN) ** GAM
        return -v * CG, -v * SG

    return h


def echo(eps: float, delay_ui: float, seg_loss_db: float):
    """One double reflection: 1 + ε exp(-α f/f_N) exp(-j 2π f τ). Not DC-normalized."""

    def h(f):
        f = _as_f(f)
        m = eps * np.exp(-seg_loss_db * NEPER_PER_DB * f / FN)
        th = -2 * np.pi * f * delay_ui * UI
        re = 1 + m * np.cos(th)
        im = m * np.sin(th)
        return 0.5 * np.log(re * re + im * im), np.arctan2(im, re)

    return h


def postcursor_echo(ratio: float, delay_ui: float = 1.0):
    """Passive two-path (1 + r exp(-j 2π f τ)) / (1 + r). Unity at DC, notch at 1/(2τ)."""

    def h(f):
        f = _as_f(f)
        th = -2 * np.pi * f * delay_ui * UI
        re = 1 + ratio * np.cos(th)
        im = ratio * np.sin(th)
        return 0.5 * np.log(re * re + im * im) - np.log1p(ratio), np.arctan2(im, re)

    return h


def ctle(gdc_db: float, gdc2_db: float):
    """IEEE 802.3ck COM CTLE.

    H(f) = (g + j f/f_z) / ((1 + j f/f_p1) (1 + j f/f_p2))
           · (g2 + j f/f_lf) / (1 + j f/f_lf)

    f_z = f_p1 = f_b/2.5, f_p2 = f_b, f_lf = f_b/80, f_b = 56 GBd.
    g = 10^(g_DC/20), g2 = 10^(g_DC2/20).
    """

    g = 10 ** (gdc_db / 20)
    g2 = 10 ** (gdc2_db / 20)
    fz = fp1 = BAUD / 2.5
    fp2 = BAUD
    flf = BAUD / 80

    def h(f):
        f = _as_f(f)
        a, b, c, d = f / fz, f / fp1, f / fp2, f / flf
        lm = 0.5 * (
            np.log(g * g + a * a)
            - np.log1p(b * b)
            - np.log1p(c * c)
            + np.log(g2 * g2 + d * d)
            - np.log1p(d * d)
        )
        ph = np.arctan2(a, g) - np.arctan(b) - np.arctan(c) + np.arctan2(d, g2) - np.arctan(d)
        return lm, ph

    return h


def channel_stages(loss_db: float, echo_ratio: float = 0.0):
    """TX driver, then the channel.

    Distributed loss at f_N is split 35% skin / 65% dielectric. A fixed echo
    (ε = 0.02, 9 UI, segment loss 0.12·loss_db) rides on every channel. The
    adjustable echo is the 1 UI two-path term.
    """

    return [
        poles(50e9, 2),
        skin(0.35 * loss_db),
        diel(0.65 * loss_db),
        echo(0.02, 9, 0.12 * loss_db),
        postcursor_echo(echo_ratio),
    ]


def rx_stages(gdc_db: float, gdc2_db: float):
    """RX termination / T-coil / ESD as one pole at 45 GHz, then the CTLE."""

    return [poles(45e9, 1), ctle(gdc_db, gdc2_db)]


def cascade(stages, f):
    """Sum of ln|H| and arg H. Each stage writes a pair of arrays."""

    f = _as_f(f)
    lm = np.zeros(f.shape, dtype=np.float64)
    ph = np.zeros(f.shape, dtype=np.float64)
    for stage in stages:
        a, b = stage(f)
        lm = lm + a
        ph = ph + b
    return lm, ph


def response_db(stages, f) -> np.ndarray:
    lm, _ = cascade(stages, f)
    return lm * DB_PER_NEPER


def pulse(stages) -> np.ndarray:
    """Unit-level 1 UI rectangle through `stages`, OS samples per UI, delayed by T0 UI.

    The rectangle's DFT (length OS, phase referenced to its first sample) is multiplied
    by H(f) exp(-j 2π f T0 UI) and inverted with a real IFFT. DC bin of the rectangle is OS.
    """

    k = np.arange(NFFT // 2 + 1)
    f = k * BAUD * OS / NFFT
    lm, ph = cascade(stages, f)
    ph = ph - 2 * np.pi * f * T0 * UI
    x = np.pi * k / NFFT
    box = np.empty(k.size, dtype=np.complex128)
    box[0] = OS
    box[1:] = np.sin(x[1:] * OS) / np.sin(x[1:]) * np.exp(-1j * x[1:] * (OS - 1))
    spectrum = np.exp(lm + 1j * ph) * box
    return np.fft.irfft(spectrum, n=NFFT)


def with_tx_ffe(p: np.ndarray, c) -> np.ndarray:
    """Three-tap TX FFE. c = (c(−1), c(0), c(+1)).

    q[n] = c(−1)·p[n+OS] + c(0)·p[n] + c(+1)·p[n−OS].
    Samples that would read outside the FFT window are left out, matching the bundle.
    """

    c = np.asarray(c, dtype=np.float64)
    q = c[1] * p
    q = np.array(q, dtype=np.float64, copy=True)
    q[:-OS] += c[0] * p[OS:]
    q[OS:] += c[2] * p[:-OS]
    return q


def conv_matrix(h: np.ndarray) -> np.ndarray:
    """Rows k = −PRE−NFPRE … POST+NFPOST, columns FFE tap a: h[k − a + NFPRE]."""

    j = KS[:, None] - np.arange(NF)[None, :] + NFPRE
    inside = (j >= -PRE) & (j <= POST)
    return np.where(inside, h[np.clip(j + PRE, 0, PRE + POST)], 0.0)


def metrics(h: np.ndarray, w: np.ndarray, nz, dfe: bool) -> dict:
    """Slicer error budget. An enabled DFE removes cursor +1 from the ISI sum.

    Parts are normalized by f0². `nz` is (th2, xt2, adc2, j2) in full-scale units.
    TX noise uses every cursor, including the one the DFE cancels.
    """

    f = conv_matrix(h) @ w
    f0 = float(f[KS == 0][0])
    f1 = float(f[KS == 1][0])
    keep = (KS != 0) if not dfe else ((KS != 0) & (KS != 1))
    isi = float(np.sum(f[keep] ** 2))
    g2 = f0 * f0 if f0 * f0 else 1e-12
    wn = float(np.dot(w, w))
    parts = {
        "isi": EA * isi / g2,
        "th": nz[0] * wn / g2,
        "xt": nz[1] * wn / g2,
        "adc": nz[2] * wn / g2,
        "jit": nz[3] * wn / g2,
        "tx": STX2 * float(np.dot(f, f)) / g2,
    }
    total = sum(parts.values())
    return {
        "f0": f0,
        "b1": (f1 / f0) if dfe and f0 else 0.0,
        "snr": EA / total,
        "parts": parts,
        "total": total,
    }


def design(h: np.ndarray, nz, dsp: bool, dfe: bool = True):
    """MMSE FFE, or a single gain 1/h[0] when DSP is off. Taps are scaled so f0 = 1.

    The normal equations use every cursor for the TX-noise term and drop cursor +1
    from the ISI Gram matrix when the DFE is on. Additive noise s2 = th+xt+adc+jit
    sits on the diagonal. The right-hand side is EA · h shifted onto the main tap.
    """

    w = np.zeros(NF, dtype=np.float64)
    if not dsp:
        w[NFPRE] = 1.0 / h[PRE]
    else:
        full = conv_matrix(h)
        m = full[KS != 1] if dfe else full
        rhs = np.zeros(NF, dtype=np.float64)
        for a in range(NF):
            j = NFPRE - a
            if -PRE <= j <= POST:
                rhs[a] = EA * h[PRE + j]
        s2 = float(sum(nz))
        gram = EA * (m.T @ m) + STX2 * (full.T @ full) + s2 * np.eye(NF)
        w = np.linalg.solve(gram, rhs)
    mtr = metrics(h, w, nz, dsp and dfe)
    w = w / mtr["f0"]
    # Recompute so b1/f0 refer to the normalized taps. SNR is unchanged.
    mtr = metrics(h, w, nz, dsp and dfe)
    return w, mtr


@dataclass
class LinkAnalysis:
    tx_ffe: bool
    dfe: bool
    echo: float
    shared_front_end: bool
    channel_loss_db: float
    gdc: float
    gdc2: float
    vga: float
    ctle_boost_db: float
    noise: tuple
    h: np.ndarray
    phase_ui: float
    ts_off: int
    adc: np.ndarray
    pad: np.ndarray
    pad_ts: int
    pad_h0: float
    pad_snr: float
    pad_range: float
    sig_pad: float
    sig_adc: float
    sig_sample: float
    weights: np.ndarray
    snr: float
    b1: float
    parts: dict


def analyze_link(
    loss_db: float = 28.0,
    xt_v: float = 1.5e-3,
    rx_noise_v: float = 0.8e-3,
    tx_ffe: bool = True,
    auto_ctle: bool = True,
    gdc: float = -17.0,
    gdc2: float = -3.0,
    dsp: bool = True,
    dfe: bool = False,
    echo: float = 0.0,
    shared_front_end: bool = True,
) -> LinkAnalysis:
    """Solve the link the way the page does on each settings change.

    Defaults are the animation's initial state (`EXAMPLE` in SerdesLink.svelte):
    28 dB distributed loss, no extra echo, TX FFE on, CTLE auto, DSP = FFE
    (DFE off), and one CTLE/clock shared across the FFE vs FFE+DFE comparison.
    """

    dfe_on = bool(dsp and dfe)
    c = TX_FFE if tx_ffe else np.array([0.0, 1.0, 0.0])
    channel = channel_stages(loss_db, echo)
    pad_pulse = with_tx_ffe(pulse(channel), c)
    gdcs = list(range(0, -21, -1)) if auto_ctle else [gdc]
    gdc2s = [0, -3, -6] if auto_ctle else [gdc2]
    df = (BAUD * OS) / NFFT
    f = np.arange(NFFT // 2 + 1) * df
    x = f / FN
    px = (x * x) / (1.0 + x**4)
    best = None
    for g in gdcs:
        for g2 in gdc2s:
            rx = rx_stages(g, g2)
            p = with_tx_ffe(pulse(channel + rx), c)
            window = p[PS : PS + 24 * OS]
            pi = PS + int(np.argmax(window))
            lm_rx, _ = cascade(rx, f)
            gg = np.exp(2.0 * lm_rx)
            cursors = p[pi + np.arange(-PRE, POST + 1) * OS]
            vga = 0.3 / np.sqrt(EA * float(np.dot(cursors, cursors)))
            th2 = (vga * rx_noise_v / VPK) ** 2 * float(np.sum(gg)) * df / FN
            xt2 = (vga * xt_v / VPK) ** 2 * float(np.sum(px * gg) / np.sum(px))
            for ts in range(pi - OS // 2, pi + OS // 2 + 1, 2):
                idx = ts + np.arange(-PRE, POST + 1) * OS
                h = p[idx] * vga
                slope = (p[idx + 1] - p[idx - 1]) * vga * OS / 2.0
                nz = (
                    th2,
                    xt2,
                    SIG_ADC**2,
                    EA * float(np.dot(slope, slope)) * (RJ / UI) ** 2,
                )
                # Matched front-end search scores the best FFE with feedback off,
                # so enabling the DFE cannot win by retuning the CTLE.
                w, mtr = design(
                    h,
                    nz,
                    shared_front_end or dsp,
                    False if shared_front_end else dfe_on,
                )
                if best is None or mtr["snr"] > best[0]:
                    best = (mtr["snr"], w, g, g2, ts, pi, vga, h, nz, p, mtr)
    if best is None:
        raise RuntimeError("no CTLE setting evaluated")
    snr, w, g, g2, ts, pi, vga, h, nz, p, mtr = best
    if shared_front_end:
        w, mtr = design(h, nz, dsp, dfe_on)
        snr = mtr["snr"]
    adc = np.empty(PLEN, dtype=np.float32)
    pad = np.empty(PLEN, dtype=np.float32)
    pad_h0 = 0.0
    pad_ts = 0
    for i in range(PLEN):
        adc[i] = np.float32(p[PS + i] * vga)
        v = float(pad_pulse[PS + i] * VPK)
        pad[i] = np.float32(v)
        if v > pad_h0:
            pad_h0 = v
            pad_ts = i
    dc = 0.0
    pr2 = 0.0
    for k in range(-PRE, POST + 1):
        i = pad_ts + k * OS
        if 0 <= i < PLEN:
            sample = float(pad[i])
            dc += sample
            pr2 += sample * sample
    sig_pad = float(np.hypot(rx_noise_v, xt_v))
    hh = float(np.dot(h, h))
    th2, xt2, adc2, j2 = nz
    ct = [ctle(g, g2)]
    channel_loss = float(-response_db(channel[1:], np.array([FN]))[0])
    boost = float(response_db(ct, np.array([FN]))[0] - response_db(ct, np.array([1e5]))[0])
    pad_snr = (EA * pad_h0**2) / (EA * max(0.0, pr2 - pad_h0**2) + STX2 * pr2 + sig_pad**2)
    pad_range = max(1.12 * abs(dc), 3.4 * np.sqrt(EA * pr2), 1.3 * pad_h0) + 3 * sig_pad
    return LinkAnalysis(
        tx_ffe=tx_ffe,
        dfe=dfe_on,
        echo=echo,
        shared_front_end=shared_front_end,
        channel_loss_db=channel_loss,
        gdc=float(g),
        gdc2=float(g2),
        vga=float(vga),
        ctle_boost_db=boost,
        noise=nz,
        h=h,
        phase_ui=(ts - pi) / OS,
        ts_off=int(ts - PS),
        adc=adc,
        pad=pad,
        pad_ts=int(pad_ts),
        pad_h0=pad_h0,
        pad_snr=float(pad_snr),
        pad_range=float(pad_range),
        sig_pad=sig_pad,
        sig_adc=float(np.sqrt(th2 + xt2 + adc2 + j2)),
        sig_sample=float(np.sqrt(th2 + xt2 + adc2 + j2 + STX2 * hh)),
        weights=w,
        snr=float(snr),
        b1=float(mtr["b1"]),
        parts=mtr["parts"],
    )


def qfunc(x: float) -> float:
    """Q(x) = erfc(x/√2)/2. The bundle's comment cites this definition.

    The browser evaluates a Numerical Recipes rational approximation. This uses
    libm erfc, which is what the page's own NumPy reference prints.
    """

    from math import erfc, sqrt

    return 0.5 * erfc(x / sqrt(2.0))


def ber_of(snr: float) -> float:
    """Gray-coded PAM4 Gaussian BER: (3/4) Q(√(SNR/5)), correct past DFE decisions."""

    from math import sqrt

    return 0.75 * qfunc(sqrt(snr / 5.0))
