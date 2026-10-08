"""Shared style for the JES figures: fonts, sizes, model colours/markers, export helper.

Typography is set for the FINAL printed size (figures are created at 180 mm width, so a
12 pt label in matplotlib is 12 pt in the journal). Colours: categorical slots validated
with the dataviz palette validator (CVD-safe adjacent pairs; low-contrast slots carry a
distinct marker shape and dark marker edge as secondary encoding).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_used"
MM = 1 / 25.4
WIDTH_MM = 180

# ---------- typography (final-size points) ----------
FS_AXIS = 13      # axis titles
FS_TICK = 11.5    # tick labels
FS_LEGEND = 11.5  # legend text
FS_NOTE = 11      # in-panel annotations
FS_PANEL = 16     # (a), (b), ...
FS_SUB = 12.5     # short panel sub-titles

INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e1e0d9"; AXIS = "#52514e"

# ---------- one colour + marker per model, used in every figure ----------
MODEL = {
    "Residual learner":                {"c": "#2a78d6", "m": "o", "ec": "#104281"},
    "ARIMA":                           {"c": "#eb6834", "m": "s", "ec": "#8a3412"},
    "ARIMA–residual average":          {"c": "#1baf7a", "m": "D", "ec": "#0b5e40"},
    "Square-root prior":               {"c": "#898781", "m": "^", "ec": "#3d3c39"},
    "ARIMA–XGBoost average (round 3)": {"c": "#4a3aa7", "m": "v", "ec": "#2a1f6b"},
    "Chronos-Bolt-small, fine-tuned":  {"c": "#e87ba4", "m": "P", "ec": "#8f2f55"},
    "Best neural, re-tuned per origin": {"c": "#008300", "m": "X", "ec": "#004d00"},
}
# ordinal blue ramp for forecast origins (validated --ordinal on white)
ORIGIN3 = ["#86b6ef", "#2a78d6", "#104281"]
ORIGIN2 = ["#6da7ec", "#184f95"]

LW = 2.5      # data lines / error bars (pt)
MS = 9.5      # marker size (pt)


def setup():
    plt.rcParams.update({
        "font.family": "Liberation Sans", "font.weight": "bold", "font.size": FS_TICK,
        "axes.labelweight": "bold", "axes.titleweight": "bold", "axes.labelsize": FS_AXIS,
        "axes.titlesize": FS_SUB, "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND, "legend.frameon": False,
        "axes.edgecolor": AXIS, "axes.linewidth": 1.4, "axes.labelcolor": INK,
        "xtick.color": INK, "ytick.color": INK, "xtick.major.width": 1.4, "ytick.major.width": 1.4,
        "xtick.major.size": 5, "ytick.major.size": 5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.linewidth": LW, "lines.markersize": MS, "errorbar.capsize": 5,
    })


def bold_ticks(ax):
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_fontweight("bold")


def panel_label(ax, text, x=-0.16, y=1.06):
    ax.text(x, y, text, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left", color=INK)


def save(fig, name):
    out = {"pdf": ROOT / "figures", "svg": ROOT / "svg", "png": ROOT / "png_600dpi"}
    for k, d in out.items():
        d.mkdir(parents=True, exist_ok=True)
    fig.savefig(out["pdf"] / f"{name}.pdf")
    fig.savefig(out["svg"] / f"{name}.svg")
    fig.savefig(out["png"] / f"{name}.png", dpi=600)
    plt.close(fig)
    print("saved", name)


# ---------- layout helpers: sub-captions BELOW panels, legends below plots ----------
FS_SUBCAP = 14   # "(a) ..." sub-caption under each panel (bold)


def _disp_to_fig(fig, x, y):
    return fig.transFigure.inverted().transform((x, y))


def lowest_y(fig, artists):
    """Lowest display-y of the tight bounding boxes of axes / legends / texts."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    return min((a.get_tightbbox(r) if hasattr(a, "get_tightbbox") else a.get_window_extent(r)).y0 for a in artists)


def subcaptions(fig, axes, labels, gap_pt=7, extra=()):
    """Place '(a) text' centred under each axis, on one common baseline below everything in the row."""
    y = lowest_y(fig, list(axes) + list(extra)) - gap_pt / 72 * fig.dpi
    texts = []
    for ax, lab in zip(axes, labels):
        bb = ax.get_window_extent(); xc = (bb.x0 + bb.x1) / 2
        fx, fy = _disp_to_fig(fig, xc, y)
        texts.append(fig.text(fx, fy, lab, ha="center", va="top", fontsize=FS_SUBCAP, fontweight="bold", color=INK))
    return texts


def legend_below(fig, artists, handles, ncol, x_center=None, gap_pt=6, **kw):
    """Figure-level legend whose top sits gap_pt below the lowest of `artists`."""
    y = lowest_y(fig, artists) - gap_pt / 72 * fig.dpi
    if x_center is None:
        x_center = fig.bbox.width / 2
    fx, fy = _disp_to_fig(fig, x_center, y)
    return fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(fx, fy), ncol=ncol, borderaxespad=0, **kw)


def note_below(fig, artists, text, gap_pt=6, x=None, ha="center"):
    y = lowest_y(fig, artists) - gap_pt / 72 * fig.dpi
    fx, fy = _disp_to_fig(fig, fig.bbox.width / 2 if x is None else x, y)
    return fig.text(fx, fy, text, ha=ha, va="top", fontsize=FS_NOTE - 0.5, fontweight="bold", color=INK2)


def check_inside(fig, artists):
    """Fail loudly if any artist is clipped by the figure edge."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); W, H = fig.bbox.width, fig.bbox.height
    for a in artists:
        b = a.get_window_extent(r)
        assert b.x0 >= -0.5 and b.y0 >= -0.5 and b.x1 <= W + 0.5 and b.y1 <= H + 0.5, (a, b, W, H)


def offset_spines(ax, pts=7):
    """Detach axes lines from the data so markers never sit on the axis lines or tick marks."""
    for s in ("left", "bottom"):
        ax.spines[s].set_position(("outward", pts))
