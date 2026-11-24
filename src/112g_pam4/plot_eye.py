#!/usr/bin/env python3
"""Rebuild the animation's Compare-tab eyes for the page's default settings.

    python src/112g_pam4/plot_eye.py
    python src/112g_pam4/plot_eye.py --loss 42 --dfe --symbols 4096 -o eye.png

The default matches the loaded page: Lossy 28 dB channel, TX FFE on, CTLE auto,
DSP = FFE, shared CTLE and clock, 1.5 mV ICN, 0.8 mV RX noise, echo 0.
The sampling eye is the rolling 4096-symbol histogram the scope settles on.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from eye import EyeStream, Receiver, colorize, composite  # noqa: E402
from model import (  # noqa: E402
    DB_PER_NEPER,
    FN,
    LEVELS,
    NFPRE,
    PRE,
    analyze_link,
    ber_of,
    channel_stages,
    design,
    diel,
    pulse,
    response_db,
    skin,
)

# Digits printed by the classroom NumPy reference and checked by the page's tests.
# Columns: loss xt_mV rxn txffe auto dsp | gdc gdc2 phase vga h-1 h0 h1 pad_mV snr_dB log10_ber b1
# The first six fields are the inputs. gdc/gdc2 in the table are the solved values.
# dfe / echo / shared_front_end are fixed by which block the row came from.
_PAGE_DEFAULT = dict(
    gdc=-17,
    gdc2=-3,
    phase=-0.06250,
    vga=9.270803,
    hm1=0.027881,
    h0=0.384238,
    h1=0.112764,
    pad_mv=56.9860,
    snr_db=20.83144,
    log10_ber=-6.4918,
    b1=0.0,
    w0=2.760330,
)


def check_page_default(a) -> None:
    """Fail if the solved default link drifts from the numbers the page displays."""

    import math

    snr_db = 10 * math.log10(a.snr)
    log10_ber = math.log10(max(ber_of(a.snr), 1e-300))
    got = {
        "gdc": a.gdc,
        "gdc2": a.gdc2,
        "phase": a.phase_ui,
        "vga": a.vga,
        "hm1": float(a.h[PRE - 1]),
        "h0": float(a.h[PRE]),
        "h1": float(a.h[PRE + 1]),
        "pad_mv": a.pad_h0 * 1e3,
        "snr_db": snr_db,
        "log10_ber": log10_ber,
        "b1": a.b1,
        "w0": float(a.weights[NFPRE]),
    }
    tol = {
        "gdc": 0,
        "gdc2": 0,
        "phase": 5e-6,
        "vga": 5e-6,
        "hm1": 5e-6,
        "h0": 5e-6,
        "h1": 5e-6,
        "pad_mv": 5e-4,
        "snr_db": 5e-5,
        "log10_ber": 5e-4,
        "b1": 5e-6,
        "w0": 5e-6,
    }
    bad = []
    for key, exp in _PAGE_DEFAULT.items():
        if abs(got[key] - exp) > tol[key]:
            bad.append(f"{key}: got {got[key]!r} expected {exp!r}")
    if bad:
        raise SystemExit("default link does not match the animation:\n  " + "\n  ".join(bad))
    # The loss hint on the page is "28.1 dB" (one decimal). Skin+dielectric alone are 28 dB;
    # the fixed 9 UI echo supplies the extra tenth.
    if abs(a.channel_loss_db - 28.1) > 0.05:
        raise SystemExit(f"channel loss {a.channel_loss_db:.4f} dB, page shows 28.1 dB")
    # FFE vs FFE+DFE on the shared front end. The page SSR prints 20.8 dB and 21.3 dB.
    ffe = design(a.h, a.noise, True, False)[1]["snr"]
    both = design(a.h, a.noise, True, True)[1]["snr"]
    ffe_db = 10 * math.log10(ffe)
    both_db = 10 * math.log10(both)
    if abs(ffe_db - 20.8) > 0.05 or abs(both_db - 21.3) > 0.05:
        raise SystemExit(f"SNR comparison {ffe_db:.2f} / {both_db:.2f} dB, page shows 20.8 / 21.3")


def check_identities() -> None:
    """Loss at f_N is the setting, and the driver+channel pulse has unit area."""

    for loss in (8.0, 28.0, 44.0):
        lm = response_db([skin(0.35 * loss), diel(0.65 * loss)], np.array([FN]))
        if abs(float(lm[0]) + loss) > 1e-9:
            raise SystemExit(f"Nyquist loss identity failed at {loss} dB ({lm[0]})")
        p = pulse([channel_stages(loss)[0], skin(0.35 * loss), diel(0.65 * loss)])
        area = float(p.sum() / OS)
        if abs(area - 1.0) > 1e-9:
            raise SystemExit(f"pulse area {area} at {loss} dB")
    # DB_PER_NEPER is the bundle's truncated constant; keep it wired so a refactor notices.
    if abs(DB_PER_NEPER - 8.685889638) > 0:
        raise SystemExit("neper constant drifted")


def _draw_eye(ax, rgb, range_, amplitude, corner, title):
    ax.imshow(rgb, extent=(-1, 1, -range_, range_), origin="upper", aspect="auto", interpolation="bilinear")
    ax.set_xlim(-1, 1)
    ax.set_ylim(-range_, range_)
    for x in np.linspace(-1, 1, 9)[1:-1]:
        ax.axvline(x, color=(23 / 255, 33 / 255, 46 / 255, 0.08), lw=0.6, zorder=2)
    for frac in (0.25, 0.5, 0.75):
        ax.axhline(range_ * (1 - 2 * frac), color=(23 / 255, 33 / 255, 46 / 255, 0.08), lw=0.6, zorder=2)
    ax.axvline(0, color=(23 / 255, 33 / 255, 46 / 255, 0.30), lw=0.8, zorder=3)
    if amplitude is not None:
        for th in (-2 / 3, 0, 2 / 3):
            ax.axhline(th * amplitude, color=(179 / 255, 105 / 255, 22 / 255, 0.75), lw=0.8, ls=(0, (3, 3)), zorder=3)
    ax.set_title(title, loc="left", fontsize=11, color="#17212e", pad=8)
    ax.text(0.012, 0.96, corner, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=(23 / 255, 33 / 255, 46 / 255, 0.72), fontfamily="monospace")
    ax.text(0.012, 0.03, "−1 UI", transform=ax.transAxes, ha="left", va="bottom", fontsize=8, color=(23 / 255, 33 / 255, 46 / 255, 0.72), fontfamily="monospace")
    ax.text(0.988, 0.03, "+1 UI", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=(23 / 255, 33 / 255, 46 / 255, 0.72), fontfamily="monospace")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d9e0e9")


def _draw_histogram(ax, counts, range_):
    n = counts.size
    edges = np.linspace(-range_, range_, n + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    peak = max(1, int(counts.max()))
    ax.bar(centres, counts, width=(edges[1] - edges[0]), color="#246dc1", alpha=0.75, align="center", linewidth=0)
    for th in (-2 / 3, 0.0, 2 / 3):
        ax.axvline(th, color="#b36916", lw=0.8, ls=(0, (3, 3)))
    ax.set_xlim(-range_, range_)
    ax.set_ylim(0, peak * 1.08)
    ax.set_xticks([-1, -1 / 3, 1 / 3, 1])
    ax.set_xticklabels(["−1", "−⅓", "+⅓", "+1"])
    ax.set_ylabel("sample count", fontsize=8, color="#55637a")
    ax.tick_params(labelsize=8, colors="#55637a")
    ax.set_title("Eye centre · histogram of the decision samples", loc="left", fontsize=11, color="#17212e", pad=8)
    ax.text(0.99, 0.95, f"peak {peak}", transform=ax.transAxes, ha="right", va="top", fontsize=8, color="#55637a", fontfamily="monospace")
    for spine in ax.spines.values():
        spine.set_color("#d9e0e9")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(analysis, stream, measured, path: Path) -> None:
    pad_rgb = composite(colorize(stream.pad_eye, light=True))
    rx_rgb = composite(colorize(stream.sampling.image, light=True))
    # Stretch the way the canvas does: both eyes share one CSS box, bilinear.
    pad_rgb = np.asarray(Image.fromarray(pad_rgb).resize((960, 300), Image.BILINEAR))
    rx_rgb = np.asarray(Image.fromarray(rx_rgb).resize((960, 300), Image.BILINEAR))
    fig = plt.figure(figsize=(11.2, 8.4), dpi=120, facecolor="#f6f8fa")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.85], hspace=0.28, wspace=0.08)
    ax0 = fig.add_subplot(grid[0, 0])
    ax1 = fig.add_subplot(grid[0, 1])
    ax2 = fig.add_subplot(grid[1, :])
    for ax in (ax0, ax1):
        ax.set_facecolor("#ffffff")
    ax2.set_facecolor("#ffffff")
    import math

    pad_db = 10 * math.log10(measured["pad_snr"]) if measured["symbols"] else float("nan")
    snr_db = 10 * math.log10(measured["snr"]) if measured["symbols"] else float("nan")
    _draw_eye(
        ax0,
        pad_rgb,
        analysis.pad_range,
        None,
        f"±{round(analysis.pad_range * 1000)} mV",
        f"Before RX EQ    SNR {pad_db:.1f} dB",
    )
    label = "After FFE + DFE" if analysis.dfe else "After CTLE + FFE"
    _draw_eye(
        ax1,
        rx_rgb,
        measured["range"],
        1.0,
        "levels ±1, ±⅓",
        f"{label}    SNR {snr_db:.1f} dB",
    )
    _draw_histogram(ax2, measured["histogram"], measured["range"])
    fig.suptitle(
        f"112G PAM4  ·  {analysis.channel_loss_db:.1f} dB at 28 GHz"
        f"  ·  g_DC {analysis.gdc:.0f} dB, g_DC2 {analysis.gdc2:.0f} dB"
        f"  ·  {measured['symbols']} decision samples",
        fontsize=12,
        color="#17212e",
        x=0.02,
        ha="left",
    )
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def level_means(stream: EyeStream) -> list[float]:
    """Mean slicer sample per transmitted level, over the symbols still in the ring."""

    sums = np.zeros(4)
    counts = np.zeros(4)
    # The ring holds the last 1024 centres. Skip the newest NFPRE, which are not decided yet.
    n_last = stream.n - 1 - NFPRE
    for n in range(n_last - 800, n_last):
        symbol = int(stream.sym[n & 1023])
        sums[symbol] += stream.centre[n & 1023]
        counts[symbol] += 1
    return [float(sums[i] / counts[i]) for i in range(4)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--loss", type=float, default=28.0, help="distributed loss at 28 GHz, dB")
    p.add_argument("--echo", type=float, default=0.0, help="1 UI echo ratio, 0 … 0.9")
    p.add_argument("--xt-mv", type=float, default=1.5, help="crosstalk ICN, mV rms")
    p.add_argument("--rx-noise-mv", type=float, default=0.8, help="RX input noise, mV rms")
    p.add_argument("--no-tx-ffe", action="store_true")
    p.add_argument("--manual-ctle", action="store_true")
    p.add_argument("--gdc", type=float, default=-17.0)
    p.add_argument("--gdc2", type=float, default=-3.0)
    p.add_argument("--dsp", choices=("off", "ffe", "dfe"), default="ffe")
    p.add_argument("--no-shared-front-end", action="store_true")
    p.add_argument("--symbols", type=int, default=4096, help="decision samples kept in the sampling eye")
    p.add_argument("-o", "--output", type=Path, default=HERE / "eye_default.png")
    args = p.parse_args(argv)

    check_identities()
    dsp = args.dsp != "off"
    dfe = args.dsp == "dfe"
    analysis = analyze_link(
        loss_db=args.loss,
        xt_v=args.xt_mv * 1e-3,
        rx_noise_v=args.rx_noise_mv * 1e-3,
        tx_ffe=not args.no_tx_ffe,
        auto_ctle=not args.manual_ctle,
        gdc=args.gdc,
        gdc2=args.gdc2,
        dsp=dsp,
        dfe=dfe,
        echo=args.echo,
        shared_front_end=not args.no_shared_front_end,
    )
    page_default = (
        args.loss == 28
        and args.echo == 0
        and args.xt_mv == 1.5
        and args.rx_noise_mv == 0.8
        and not args.no_tx_ffe
        and not args.manual_ctle
        and args.dsp == "ffe"
        and not args.no_shared_front_end
    )
    if page_default:
        check_page_default(analysis)

    # Warmup inside EyeStream is 43 symbols before the first recorded decision
    # (recording starts when warmup reaches NF+32 = 44). Ask for `symbols` recorded.
    rx = Receiver(analysis)
    stream = EyeStream(analysis, rx)
    stream.run(43 + args.symbols)
    measured = stream.measurements()
    if measured["symbols"] != args.symbols:
        raise SystemExit(f"expected {args.symbols} decision samples, got {measured['symbols']}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_figure(analysis, stream, measured, args.output)
    means = level_means(stream)
    import math

    print(f"channel loss at 28 GHz : {analysis.channel_loss_db:.4f} dB")
    print(f"CTLE g_DC / g_DC2      : {analysis.gdc:.0f} / {analysis.gdc2:.0f} dB")
    print(f"CTLE boost             : {analysis.ctle_boost_db:.2f} dB")
    print(f"VGA                    : {analysis.vga:.6f}  ({20 * math.log10(analysis.vga):.2f} dB)")
    print(f"sampling phase         : {analysis.phase_ui:.5f} UI from the pulse peak")
    print(f"model SNR              : {10 * math.log10(analysis.snr):.4f} dB")
    print(f"model BER              : {ber_of(analysis.snr):.3e}")
    print(f"DFE b1                 : {analysis.b1:.5f}")
    print(f"measured SNR           : {10 * math.log10(measured['snr']):.2f} dB over {measured['symbols']} symbols")
    print(f"measured pad SNR       : {10 * math.log10(measured['pad_snr']):.2f} dB")
    print(f"observed errors / bits : {measured['bit_errors']} / {measured['bits']}")
    print("slicer means           : " + "  ".join(f"{m:+.3f}" for m in means))
    print(f"eye amplitude range    : ±{measured['range']}")
    print(f"wrote {args.output}")
    # The equalized outer levels are ±1 and the inner levels ±1/3. A mis-wired
    # cursor or a missing VGA shows up here immediately.
    if page_default and dsp:
        for got, exp in zip(means, LEVELS):
            if abs(got - float(exp)) > 0.08:
                raise SystemExit(f"slicer mean {got:.3f} is far from level {exp:.3f}")
        if measured["bit_errors"] != 0:
            raise SystemExit(f"default eye still has {measured['bit_errors']} bit errors; the opening should be clean")


if __name__ == "__main__":
    main()
