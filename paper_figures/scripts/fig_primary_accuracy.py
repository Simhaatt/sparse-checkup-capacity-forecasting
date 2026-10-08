"""Figure 3 - primary forecasting accuracy on the 104-cell Luh-Blank cohort (origins 10/15/20).

Data: data_used/fig3_primary_accuracy.csv (means and 95 % cell-bootstrap CIs from r4_summary.csv;
fine-tuned Chronos CIs bootstrapped from its saved predictions by prepare_plot_data.py with the same
procedure) and data_used/fig3_holm_vs_arima.csv (pre-declared Holm families r4_vs_ARIMA_origin{o}).
Layout: plot -> axis title -> legend -> sub-caption "(a) ..." below each panel.
Run:  python fig_primary_accuracy.py
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jes_style as S

S.setup()
d = pd.read_csv(S.DATA / "fig3_primary_accuracy.csv")
holm = pd.read_csv(S.DATA / "fig3_holm_vs_arima.csv")
ORIG = [10, 15, 20]
PANELS = {
    "a": ["Square-root prior", "ARIMA", "Residual learner", "ARIMA–residual average"],
    "b": ["Residual learner", "ARIMA–XGBoost average (round 3)", "Chronos-Bolt-small, fine-tuned", "Best neural, re-tuned per origin"],
}
SUBCAP = {"a": "(a) Principal methods", "b": "(b) Additional comparators"}
LEG = {"ARIMA–XGBoost average (round 3)": "ARIMA–XGB average (round 3)", "Residual learner": "Residual learner (ours)",
       "ARIMA–residual average": "ARIMA–residual average (ours)"}
sig = set()   # only pre-declared Holm-adjusted p < 0.05 versus ARIMA
for _, r in holm.iterrows():
    if r.p_holm < 0.05:
        sig.add(("ARIMA–residual average" if r.comparison.startswith("avg_ARIMA_hybrid") else "Residual learner", int(r.origin)))

fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_MM * S.MM, 124 * S.MM), sharey=True)
fig.subplots_adjust(left=0.095, right=0.99, top=0.975, bottom=0.47, wspace=0.13)
legends = []
for ax, (key, methods) in zip(axes, PANELS.items()):
    S.offset_spines(ax)
    offs = np.linspace(-0.27, 0.27, len(methods)); handles = []
    for off, m in zip(offs, methods):
        st = S.MODEL[m]; g = d[d.label.eq(m)].set_index("origin")
        os_ = [o for o in ORIG if o in g.index]; xs = [ORIG.index(o) + off for o in os_]
        y = g.loc[os_, "MAPE"].to_numpy(); lo = g.loc[os_, "ci_low"].to_numpy(); hi = g.loc[os_, "ci_high"].to_numpy()
        ax.errorbar(xs, y, yerr=[y - lo, hi - y], fmt=st["m"], color=st["c"], mfc=st["c"], mec=st["ec"], mew=1.4,
                    ms=S.MS, elinewidth=S.LW, capsize=5, capthick=S.LW, zorder=3)
        if key == "a":
            for x, o, h in zip(xs, os_, hi):
                if (m, o) in sig:
                    ax.text(x, h + 0.10, "*", ha="center", va="bottom", fontsize=17, fontweight="bold", color=S.INK)
        if key == "b" and m == "Best neural, re-tuned per origin":
            ax.text(2 + off, 0.35, "not\nrun", ha="center", va="bottom", fontsize=S.FS_NOTE - 1, color=S.INK2, fontweight="bold", linespacing=0.95)
        lab = "Residual learner (ours, reference)" if (key == "b" and m == "Residual learner") else LEG.get(m, m)
        handles.append(Line2D([], [], color=st["c"], marker=st["m"], mec=st["ec"], mew=1.4, ms=S.MS, lw=S.LW, label=lab))
    ax.set_xticks(range(3)); ax.set_xticklabels([str(o) for o in ORIG])
    ax.set_xlim(-0.55, 2.55); ax.set_ylim(0, 6.3); ax.set_yticks(range(0, 7))
    ax.set_xlabel("Forecast origin (checkups)")
    S.bold_ticks(ax)
    ax._jes_handles = handles
axes[0].set_ylabel("Capacity MAPE (%)")
# legends directly under each panel, then the sub-captions on one common line, then the note
for ax in axes:
    bb = ax.get_window_extent(); xc = (bb.x0 + bb.x1) / 2
    legends.append(S.legend_below(fig, list(axes), ax._jes_handles, ncol=1, x_center=xc, gap_pt=6,
                                  handlelength=2.2, handletextpad=0.5, labelspacing=0.3))
caps = S.subcaptions(fig, axes, [SUBCAP["a"], SUBCAP["b"]], gap_pt=8, extra=legends)
note = S.note_below(fig, caps, "Error bars: 95 % cell-bootstrap CIs (104 cells).  * Holm-adjusted p < 0.05 versus ARIMA.", gap_pt=8)
S.check_inside(fig, legends + caps + [note])
S.save(fig, "fig_primary_accuracy")
