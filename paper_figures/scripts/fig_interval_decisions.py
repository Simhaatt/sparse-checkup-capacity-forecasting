"""Figure 6 - prediction intervals and 80 %-capacity crossing decisions for the residual learner.

Data (all in data_used/, copied unchanged from the round-4 tables):
  fig6_r4_hybrid_conformal_luh.csv      (a) split-conformal coverage, 20 independent calibration cells/fold
  fig6_r4_g5_mondrian_by_regime.csv     (b) regime-stratified (Mondrian) split-conformal, 80 % nominal
  fig6_r4_hybrid_conformal_snl.csv      (c) Sandia cross-fold (CV+-style) residual intervals, 80 % nominal
  fig6_sandia_cells_by_chemistry.csv    (c) number of scored cells per chemistry
  fig6_r4_decisions_luh_r3rule.csv      (d) crossing F1, round-3 eligibility rule (corrected file)
Run:  python fig_interval_decisions.py
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
import jes_style as S

S.setup()
L = pd.read_csv(S.DATA / "fig6_r4_hybrid_conformal_luh.csv")
M = pd.read_csv(S.DATA / "fig6_r4_g5_mondrian_by_regime.csv")
N = pd.read_csv(S.DATA / "fig6_r4_hybrid_conformal_snl.csv")
NC = pd.read_csv(S.DATA / "fig6_sandia_cells_by_chemistry.csv").set_index(["origin", "chem"]).n_cells
D = pd.read_csv(S.DATA / "fig6_r4_decisions_luh_r3rule.csv")

fig, ax = plt.subplots(2, 2, figsize=(S.WIDTH_MM * S.MM, 214 * S.MM))
fig.subplots_adjust(left=0.095, right=0.985, top=0.985, bottom=0.215, wspace=0.34, hspace=0.86)
LEGS = {}; NOTES = []
REF = dict(color=S.INK2, lw=1.8, ls=(0, (5, 3)), zorder=1)


SUBCAPS = {}


def head(a, k, title, top=1.0):
    SUBCAPS[k] = (a, f"({k}) {title}")
    a.set_ylim(0, top); a.set_yticks(np.arange(0, 1.01, 0.2)); S.bold_ticks(a)


def under_legend(a, handles, ncol, **kw):
    bb = a.get_window_extent()
    LEGS[a] = (handles, ncol, (bb.x0 + bb.x1) / 2, kw)

# (a) split-conformal coverage by origin, 80 % and 90 % nominal
a = ax[0, 0]; c = S.MODEL["Residual learner"]
O = [10, 15, 20]; x = np.arange(3)
for nom, mfc, dx in ((0.8, c["c"], -0.12), (0.9, "white", 0.12)):
    r = L[L.nominal.eq(nom)].set_index("origin").loc[O]
    a.plot(x + dx, r.coverage_equal_cell, ls="none", marker="o", ms=11, mfc=mfc, mec=c["ec"], mew=2.2, zorder=3,
           path_effects=[pe.Stroke(linewidth=5, foreground="white"), pe.Normal()])
    a.axhline(nom, **REF)
a.set_xticks(x); a.set_xticklabels(O); a.set_xlim(-0.5, 2.5); a.set_xlabel("Forecast origin (checkups)"); a.set_ylabel("Empirical coverage")
S.offset_spines(a)
head(a, "a", "Split-conformal coverage")
under_legend(a, [Line2D([], [], ls="none", marker="o", ms=10, mfc=c["c"], mec=c["ec"], mew=2, label="80 % interval"),
                 Line2D([], [], ls="none", marker="o", ms=10, mfc="white", mec=c["ec"], mew=2, label="90 % interval"),
                 Line2D([], [], **{k: v for k, v in REF.items() if k != "zorder"}, label="Nominal level")], ncol=2)

# (b) regime-stratified coverage, 80 % nominal, bars coloured by origin (ordinal ramp)
a = ax[0, 1]; REG = [(1, "Calendar"), (2, "Cyclic"), (3, "Profile")]
w = 0.26
for j, o in enumerate(O):
    r = M[(M.nominal == 0.8) & (M.origin == o)].set_index("age_type").loc[[g for g, _ in REG]]
    a.bar(np.arange(3) + (j - 1) * (w + 0.02), r.coverage, width=w, color=S.ORIGIN3[j], edgecolor="#0d366b", linewidth=1.0, zorder=2)
a.axhline(0.8, **REF)
a.set_xticks(range(3)); a.set_xticklabels([n for _, n in REG]); a.set_xlim(-0.55, 2.55); a.set_xlabel("Ageing regime"); a.set_ylabel("Coverage, 80 % nominal")
head(a, "b", "Coverage by ageing regime")
under_legend(a, [Patch(facecolor=S.ORIGIN3[j], edgecolor="#0d366b", label=f"Origin {o}") for j, o in enumerate(O)]
             + [Line2D([], [], **{k: v for k, v in REF.items() if k != "zorder"}, label="Nominal 80 %")], ncol=2)

# (c) Sandia cross-fold residual intervals, 80 % nominal, by chemistry
a = ax[1, 0]; CH = ["LFP", "NCA", "NMC"]; w = 0.34
for j, o in enumerate([5, 8]):
    r = N[(N.nominal == 0.8) & (N.origin == o)].iloc[0]
    xs = np.arange(3) + (j - 0.5) * (w + 0.03)
    a.bar(xs, [r[ch] for ch in CH], width=w, color=S.ORIGIN2[j], edgecolor="#0d366b", linewidth=1.0, zorder=2)
    for xx, ch in zip(xs, CH):
        a.text(xx, 0.035, f"n={NC[(o, ch)]}", ha="center", va="bottom", fontsize=S.FS_NOTE - 0.5, fontweight="bold",
               color=S.INK if j == 0 else "white", rotation=90, zorder=4)
a.axhline(0.8, **REF)
a.set_xticks(range(3)); a.set_xticklabels(CH); a.set_xlim(-0.55, 2.55); a.set_xlabel("Chemistry (Sandia)"); a.set_ylabel("Coverage, 80 % nominal")
head(a, "c", "Sandia, cross-fold intervals")
under_legend(a, [Patch(facecolor=S.ORIGIN2[j], edgecolor="#0d366b", label=f"Origin {o}") for j, o in enumerate([5, 8])]
             + [Line2D([], [], **{k: v for k, v in REF.items() if k != "zorder"}, label="Nominal 80 %")], ncol=2)

# (d) F1 for detecting the first crossing below 80 % of initial capacity
a = ax[1, 1]; w = 0.36
for j, (meth, name) in enumerate((("ARIMA_AICc", "ARIMA"), ("hybrid_sqrt_xgb", "Residual learner"))):
    st = S.MODEL[name]; r = D[D.method.eq(meth)].set_index("origin").loc[O]
    xs = np.arange(3) + (j - 0.5) * (w + 0.03)
    a.bar(xs, r.F1, width=w, color=st["c"], edgecolor=st["ec"], linewidth=1.2, zorder=2)
    for xx, v in zip(xs, r.F1):
        a.text(xx, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=S.FS_NOTE, fontweight="bold", color=S.INK)
a.set_xticks(range(3)); a.set_xticklabels(O); a.set_xlim(-0.55, 2.55); a.set_xlabel("Forecast origin (checkups)"); a.set_ylabel("Crossing-detection F1")
head(a, "d", "80 % capacity crossing", top=1.1)
ev = D[D.method.eq("ARIMA_AICc")].set_index("origin").loc[O]
under_legend(a, [Patch(facecolor=S.MODEL[n]["c"], edgecolor=S.MODEL[n]["ec"], label=lab) for n, lab in (("ARIMA", "ARIMA"), ("Residual learner", "Residual learner"))], ncol=2)
CROSS = "Crossings / eligible cells:\n" + ", ".join(f"{int(e)}/{int(n)}" for e, n in zip(ev.true_events, ev.eligible))
# ---------- layout: legend under each panel, then sub-caption under it (row by row) ----------
fig.canvas.draw()
allcaps = []
for row in (0, 1):
    row_axes = [ax[row, 0], ax[row, 1]]
    legs = []
    for a in row_axes:
        handles, ncol, xc, kw = LEGS[a]
        legs.append(S.legend_below(fig, row_axes, handles, ncol, x_center=xc, gap_pt=6, handlelength=1.8, columnspacing=1.0, handletextpad=0.4, **kw))
    extra = list(legs)
    if row == 1:
        bb = ax[1, 1].get_window_extent()
        extra.append(S.note_below(fig, [legs[1]], CROSS, gap_pt=4, x=(bb.x0 + bb.x1) / 2))
    keys = ["a", "b"] if row == 0 else ["c", "d"]
    allcaps += S.subcaptions(fig, row_axes, [SUBCAPS[k][1] for k in keys], gap_pt=8, extra=extra)
    allcaps += extra
    if row == 0:
        row0_items = list(allcaps)
# top-row sub-captions must clear the bottom row
fig.canvas.draw(); r = fig.canvas.get_renderer()
top_of_row2 = max(a.get_tightbbox(r).y1 for a in (ax[1, 0], ax[1, 1]))
gap = min(t.get_window_extent(r).y0 for t in row0_items) - top_of_row2
print('row gap px', round(gap, 1))
assert gap > 4, "rows overlap"
S.check_inside(fig, allcaps)
S.save(fig, "fig_interval_decisions")
