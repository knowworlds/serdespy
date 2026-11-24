"""Eye accumulation and drawing from the animation's streams.ts and eyes.ts.

Two eyes are what the Compare tab shows:

* Before RX EQ — 2 UI waveform cuts at the RX pad, rasterized into a density image.
* After CTLE + FFE — one amplitude histogram per clock phase. The measured UI is
  painted twice so the canvas is 2 UI wide. Column OS (the centre) is the slicer
  sample; the histogram under the eyes is that column.

DFE, when enabled, is applied only at the decision phases of the sampling eye.
Each phase keeps its own previous decision. The continuous FFE trace is not a
DFE-corrected waveform; the page says so and this module follows that.
"""

from __future__ import annotations

from math import ceil, floor, sqrt

import numpy as np

from model import (
    EA,
    LEVELS,
    NF,
    NFPRE,
    OS,
    STX2,
    LinkAnalysis,
    metrics,
)

# Ring length in streams.ts. Symbol index n is stored at n & MASK.
RING = 1024
MASK = RING - 1

# Light-theme persistence ramp from eyes.ts. Alpha 0 at empty bins, so the
# plot background shows through. Stops are (position, R, G, B, A).
_LIGHT_STOPS = (
    (0.0, (200, 222, 245, 0)),
    (0.1, (150, 190, 235, 150)),
    (0.35, (60, 120, 200, 235)),
    (0.65, (30, 80, 160, 255)),
    (1.0, (10, 30, 70, 255)),
)
_DARK_STOPS = (
    (0.0, (20, 50, 95, 0)),
    (0.1, (32, 84, 150, 150)),
    (0.35, (70, 145, 235, 235)),
    (0.65, (150, 200, 255, 255)),
    (1.0, (245, 250, 255, 255)),
)

MEASURE = 4096


def _uint32(x: int) -> int:
    return x & 0xFFFFFFFF


def _to_int32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x >= 0x80000000 else x


def _imul(a: int, b: int) -> int:
    return _to_int32(_to_int32(a) * _to_int32(b))


def mulberry32(seed: int):
    """The page's mulberry32 (`rng.ts` / `_astro/rng.9Tp-K6lW.js`). Returns U[0, 1)."""

    s = _uint32(seed)

    def rnd() -> float:
        nonlocal s
        s = _uint32(s + 0x6D2B79F5)
        t = _imul(s ^ (s >> 15), s | 1)
        prod = _imul(_uint32(t) ^ (_uint32(t) >> 7), _uint32(t) | 61)
        t = _to_int32(t) ^ _to_int32(_to_int32(t) + _to_int32(prod))
        mixed = _uint32(t) ^ (_uint32(t) >> 14)
        return mixed / 4294967296

    return rnd


def normals(seed: int):
    """Box–Muller pair, spare sample retained, matching streams.ts `normals`."""

    rnd = mulberry32(seed)
    spare = None

    def gauss() -> float:
        nonlocal spare
        if spare is not None:
            s = spare
            spare = None
            return s
        u = rnd() or 1e-12
        th = 2 * np.pi * rnd()
        r = sqrt(-2 * np.log(u))
        spare = r * np.sin(th)
        return r * np.cos(th)

    return gauss


class Prbs13:
    """PRBS13, polynomial x^13 + x^12 + x^2 + x + 1, Gray-mapped to PAM4 indices.

    symbol() bit order is MSB then LSB:
    00 → 0 (−3), 01 → 1 (−1), 11 → 2 (+1), 10 → 3 (+3).
    The waveform levels are LEVELS[index], i.e. −1, −1/3, +1/3, +1, not ±3.
    """

    def __init__(self, seed: int):
        self.s = seed & 0x1FFF or 1

    def bit(self) -> int:
        s = self.s
        b = ((s >> 12) ^ (s >> 11) ^ (s >> 1) ^ s) & 1
        self.s = ((s << 1) | b) & 0x1FFF
        return b

    def symbol(self) -> int:
        msb, lsb = self.bit(), self.bit()
        if msb:
            return 2 if lsb else 3
        return 1 if lsb else 0


class Receiver:
    """FFE taps. The static eye uses the MMSE taps directly (no adaptation)."""

    def __init__(self, analysis: LinkAnalysis):
        self.a = analysis
        self.taps = np.array(analysis.weights, dtype=np.float64, copy=True)
        self.live = metrics(analysis.h, self.taps, analysis.noise, analysis.dfe)


class EyeImage:
    """Density image. Row 0 is the top of the eye (+range)."""

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buf = np.zeros(width * height, dtype=np.float64)

    def trace(self, win: np.ndarray, range_: float) -> None:
        """Rasterize one polyline. Weight of a segment is 1/√(steps), matching EyeImage.trace."""

        w = self.width
        h = self.height
        last = win.shape[0] - 1
        sx = (w - 1) / last
        sy = (h - 1) / (2 * range_)
        x0 = 0.0
        y0 = (range_ - float(win[0])) * sy
        buf = self.buf
        for i in range(1, last + 1):
            x1 = i * sx
            y1 = (range_ - float(win[i])) * sy
            dx = x1 - x0
            dy = y1 - y0
            steps = max(1, ceil(max(abs(dx), abs(dy))))
            weight = 1.0 / sqrt(steps)
            inv = 1.0 / steps
            for s in range(steps):
                t = s * inv
                yi = floor(y0 + dy * t)
                if 0 <= yi < h:
                    buf[yi * w + floor(x0 + dx * t)] += weight
            x0, y0 = x1, y1


class SamplingEye:
    """One histogram per phase, over a rolling window of MEASURE symbols.

    Width is 2·OS+1. Phase p is written to columns p and p+OS; phase 0 is also
    written to column 2·OS, so the decision sample sits on the centre line and
    on both edges. Amplitude bins span ±range. range starts at 1.6 and only
    grows, in steps of ceil(|v|·1.05·5)/5.
    """

    def __init__(self):
        self.image = EyeImage(2 * OS + 1, 160)
        self.range = 1.6
        self.count = 0
        self._next = 0
        self._values = np.zeros(MEASURE * OS, dtype=np.float64)

    def _accumulate(self, offset: int, weight: float) -> None:
        width = self.image.width
        height = self.image.height
        buf = self.image.buf
        span = self.range
        vals = self._values
        for phase in range(OS):
            raw = (vals[offset + phase] / span + 1.0) * 0.5 * height
            bin_ = min(height - 1, max(0, floor(raw)))
            row = (height - 1 - bin_) * width
            buf[row + phase] += weight
            buf[row + phase + OS] += weight
            if phase == 0:
                buf[row + 2 * OS] += weight

    def _needed(self, values) -> float:
        needed = 1.6
        for v in values:
            a = abs(float(v))
            if a >= needed:
                needed = ceil(a * 1.05 * 5) / 5
        return needed

    def add(self, phases: np.ndarray) -> None:
        needed = self.range
        for v in phases:
            a = abs(float(v))
            if a >= needed:
                needed = ceil(a * 1.05 * 5) / 5
        if needed > self.range:
            self.range = needed
            self.image.buf.fill(0.0)
            for i in range(self.count):
                self._accumulate(i * OS, 1.0)
        offset = self._next * OS
        if self.count == MEASURE:
            self._accumulate(offset, -1.0)
        else:
            self.count += 1
        self._values[offset : offset + OS] = phases
        self._accumulate(offset, 1.0)
        self._next = (self._next + 1) % MEASURE

    def snapshot(self):
        needed = self._needed(self._values[: self.count * OS])
        if needed < self.range:
            self.range = needed
            self.image.buf.fill(0.0)
            for i in range(self.count):
                self._accumulate(i * OS, 1.0)
        height = self.image.height
        width = self.image.width
        counts = np.empty(height, dtype=np.int64)
        buf = self.image.buf
        for bin_ in range(height):
            counts[bin_] = int(buf[(height - 1 - bin_) * width + OS])
        return counts


def _tap_matrix(q: np.ndarray, phase0: int):
    """Weights W[i, j] so a 65-sample window is W @ ring[n - lags].

    Sample i of the window sits at pulse-slice index (phase0 + i) + d·OS.
    q is the kept float32 slice; products happen in float64 after the load,
    which is what the JS engine does when it multiplies two Float32 values.
    """

    q64 = np.asarray(q, dtype=np.float64)
    phase = phase0 + np.arange(2 * OS + 1)
    d_lo = int(np.ceil(-phase.max() / OS))
    d_hi = int(np.floor((q64.size - 1 - phase.min()) / OS))
    lags = np.arange(d_lo, d_hi + 1)
    k = lags[None, :] * OS + phase[:, None]
    valid = (k >= 0) & (k < q64.size)
    w = np.zeros(k.shape, dtype=np.float64)
    w[valid] = q64[k[valid]]
    return lags, w


class EyeStream:
    """Same symbol path as EyeStream in streams.ts, seed 0x0b5 unless overridden.

    `run(count)` advances `count` symbols. The sampling eye starts recording once
    44 symbols have warmed the FFE delay line and the per-phase DFE history
    (NF + 32 in the source).
    """

    def __init__(self, analysis: LinkAnalysis, rx: Receiver, seed: int = 0x0B5):
        self.a = analysis
        self.rx = rx
        self.pad_eye = EyeImage(256, 160)
        self.sampling = SamplingEye()
        self.prbs = Prbs13(seed)
        self.gauss = normals(seed * 104729 + 3)
        self.sym = np.zeros(RING, dtype=np.uint8)
        self.value = np.zeros(RING, dtype=np.float32)
        self.decision = np.zeros(RING, dtype=np.uint8)
        self.centre = np.zeros(RING, dtype=np.float64)
        self.pad_centre = np.zeros(RING, dtype=np.float64)
        self.phase_decisions = np.zeros(OS, dtype=np.uint8)
        self.n = 64
        self.generated = 0
        self.warmup = 0
        self.measured = 0
        self.error_power = 0.0
        self.pad_error_power = 0.0
        self.bit_errors = 0
        self._residuals = np.zeros(MEASURE, dtype=np.float64)
        self._pad_residuals = np.zeros(MEASURE, dtype=np.float64)
        self._wrong_bits = np.zeros(MEASURE, dtype=np.int16)
        self._pad_win = [np.zeros(2 * OS + 1, dtype=np.float64) for _ in range(16)]
        self._adc_win = [np.zeros(2 * OS + 1, dtype=np.float64) for _ in range(16)]
        # phase0 = padTs − OS, so window i of symbol n is at m = n·OS + PS + phase0 + i.
        self._pad_lags, self._pad_w = _tap_matrix(analysis.pad, analysis.pad_ts - OS)
        self._adc_lags, self._adc_w = _tap_matrix(analysis.adc, analysis.ts_off - OS)

    def _generate_until(self, upto: int) -> None:
        stx = sqrt(STX2)
        while self.generated <= upto:
            s = self.prbs.symbol()
            i = self.generated & MASK
            self.sym[i] = s
            self.value[i] = np.float32(LEVELS[s] + self.gauss() * stx)
            self.generated += 1

    def _fold(self, lags: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
        idx = (n - lags) & MASK
        return weights @ self.value[idx].astype(np.float64)

    def run(self, count: int) -> None:
        a = self.a
        taps = self.rx.taps
        f0 = self.rx.live["f0"] or 1.0
        b1 = self.rx.live["b1"]
        levels = LEVELS
        for _ in range(count):
            n = self.n
            self.n = n + 1
            self._generate_until(n + 34)
            wp = self._pad_win[n & 15]
            wa = self._adc_win[n & 15]
            first = (OS + 1) if self.warmup else 0
            if self.warmup:
                prev_p = self._pad_win[(n - 1) & 15]
                prev_a = self._adc_win[(n - 1) & 15]
                wp[: OS + 1] = prev_p[OS:]
                wa[: OS + 1] = prev_a[OS:]
            pad_full = self._fold(self._pad_lags, self._pad_w, n)
            adc_full = self._fold(self._adc_lags, self._adc_w, n)
            sig_pad = a.sig_pad
            sig_adc = a.sig_adc
            g = self.gauss
            for i in range(first, 2 * OS + 1):
                wp[i] = pad_full[i] + g() * sig_pad
                wa[i] = min(1.0, max(-1.0, adc_full[i] + g() * sig_adc))
            self.pad_centre[n & MASK] = wp[OS]
            self.pad_eye.trace(wp, a.pad_range)
            self.warmup += 1
            if self.warmup < NF:
                continue
            nd = n - NFPRE
            ffe = np.zeros(2 * OS + 1, dtype=np.float64)
            for t in range(NF):
                ffe += taps[t] * self._adc_win[(nd - t + NFPRE) & 15]
            ffe /= f0
            feedback = b1 * levels[self.decision[(nd - 1) & MASK]]
            sample = float(ffe[OS] - feedback)
            self.centre[nd & MASK] = sample
            decided = 0 if sample < -2 / 3 else 1 if sample < 0 else 2 if sample < 2 / 3 else 3
            self.decision[nd & MASK] = decided
            phases = np.empty(OS, dtype=np.float64)
            phases[0] = sample
            self.phase_decisions[0] = decided
            for phase in range(1, OS):
                z = float(ffe[OS + phase] - b1 * levels[self.phase_decisions[phase]])
                phases[phase] = z
                self.phase_decisions[phase] = (
                    0 if z < -2 / 3 else 1 if z < 0 else 2 if z < 2 / 3 else 3
                )
            if self.warmup < NF + 32:
                continue
            self.sampling.add(phases)
            index = self.measured % MEASURE
            self.measured += 1
            symbol = int(self.sym[nd & MASK])
            decoded = decided
            residual = (phases[0] - levels[symbol]) ** 2
            pad_residual = (self.pad_centre[nd & MASK] / a.pad_h0 - levels[symbol]) ** 2
            gray = (symbol ^ (symbol >> 1)) ^ (decoded ^ (decoded >> 1))
            wrong = (gray & 1) + ((gray >> 1) & 1)
            self.error_power += residual - self._residuals[index]
            self.pad_error_power += pad_residual - self._pad_residuals[index]
            self.bit_errors += wrong - int(self._wrong_bits[index])
            self._residuals[index] = residual
            self._pad_residuals[index] = pad_residual
            self._wrong_bits[index] = wrong

    def measurements(self) -> dict:
        count = min(MEASURE, self.measured)
        counts = self.sampling.snapshot()
        return {
            "symbols": count,
            "bits": 2 * count,
            "bit_errors": int(self.bit_errors),
            "snr": (EA * count / max(1e-20, self.error_power)) if count else 0.0,
            "pad_snr": (EA * count / max(1e-20, self.pad_error_power)) if count else 0.0,
            "histogram": counts,
            "range": self.sampling.range,
        }


def _clamp_u8(values: np.ndarray) -> np.ndarray:
    """ECMAScript ToUint8Clamp: round half to even, then saturate."""

    x = np.asarray(values, dtype=np.float64)
    floor_x = np.floor(x)
    frac = x - floor_x
    up = frac > 0.5
    down = frac < 0.5
    half = ~(up | down)
    even = (floor_x.astype(np.int64) & 1) == 0
    rounded = np.where(up, floor_x + 1, np.where(down, floor_x, np.where(even, floor_x, floor_x + 1)))
    return np.clip(rounded, 0, 255).astype(np.uint8)


def color_lut(light: bool = True) -> np.ndarray:
    stops = _LIGHT_STOPS if light else _DARK_STOPS
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        x = i / 255
        k = 0
        while k < len(stops) - 2 and x > stops[k + 1][0]:
            k += 1
        xa, ca = stops[k][0], stops[k][1]
        xb, cb = stops[k + 1][0], stops[k + 1][1]
        t = (x - xa) / (xb - xa)
        lut[i] = _clamp_u8([ca[j] + (cb[j] - ca[j]) * t for j in range(4)])
    return lut


def colorize(image: EyeImage, light: bool = True) -> np.ndarray:
    """Map density through √(min(1, 1.7·bin/max)) into the persistence LUT.

    Returns RGBA with row 0 at the top. Empty bins keep alpha 0.
    """

    lut = color_lut(light)
    buf = image.buf.reshape(image.height, image.width)
    peak = float(buf.max())
    inv = 1.7 / peak if peak > 0 else 0.0
    idx = (np.sqrt(np.minimum(1.0, buf * inv)) * 255).astype(np.int64)
    return lut[idx]


def composite(rgba: np.ndarray, background=(255, 255, 255)) -> np.ndarray:
    """Flatten the eye onto an opaque background. Canvas clears to --plot (#fff)."""

    rgb = rgba[:, :, :3].astype(np.float64)
    a = rgba[:, :, 3:4].astype(np.float64) / 255.0
    bg = np.array(background, dtype=np.float64)
    return np.clip(rgb * a + bg * (1.0 - a), 0, 255).astype(np.uint8)
