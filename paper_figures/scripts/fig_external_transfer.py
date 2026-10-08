"""Figure 5 - cross-format transfer: Kirkaldy 21700 cylindrical (40 cells) and Oxford pouch (8 cells).

Data: data_used/fig5_transfer.csv. Means equal r4_transfer_summary.csv and r4_g1_oxford_summary.csv
(asserted in prepare_plot_data.py); error bars are 95 % cell-bootstrap CIs computed from the saved
predictions with the round-4 procedure. The residual learner is the frozen unit-free variant trained
without any target-dataset records; each target cell still supplies its own measured prefix.
Run:  python fig_external_transfer.py
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.patches import Patch
import jes_style as S

S.setup()
plt.rcParams["hatch.linewidth"] = 1.2
d = pd.read_csv(S.DATA / "fig5_transfer.csv")
ORDER = [("ARIMA_AICc", "ARIMA", "ARIMA", None),
         ("sqrt_idx_prior", "Square-root prior", "Square-root prior (index)", "//"),
         ("residual", "Residual learner", "Residual learner (ours, frozen)", None)]
d["key"] = d.method.replace({"hybrid_transfer": "residual", "hybrid_frozen": "residual"})
PANELS = [("Kirkaldy 21700 (40 cells)", "Cylindrical 21700 cells (n = 40)", [3, 5, 8], 13),
          ("Oxford pouch (8 cells)", "Pouch cells (n = 8)", [5, 10, 20], 17)]

fig, axes = plt.subplots(1, 2, figsize=(S.WIDTH_MM * S.MM, 106 * S.MM))
fig.subplots_adjust(left=0.095, right=0.99, top=0.975, bottom=0.37, wspace=0.20)
w = 0.26
for k, (ax, (ds, title, origins, ymax)) in enumerate(zip(axes, PANELS)):
    g = d[d.dataset.eq(ds)]
    for j, (key, model, _, hatch) in enumerate(ORDER):
        st = S.MODEL[model]; r = g[g.key.eq(key)].set_index("origin").loc[origins]
        x = np.arange(len(origins)) + (j - 1) * (w + 0.02)
        ax.bar(x, r.MAPE, width=w, color=st["c"], edgecolor=st["ec"], linewidth=1.2, hatch=hatch, zorder=2)
        ax.errorbar(x, r.MAPE, yerr=[r.MAPE - r.ci_low, r.ci_high - r.MAPE], fmt="none", ecolor=S.INK, elinewidth=2.0, capsize=4, capthick=2.0, zorder=3)
    ax.set_xticks(range(len(origins))); ax.set_xticklabels([str(o) for o in origins])
    ax.set_xlim(-0.6, len(origins) - 0.4); ax.set_ylim(0, ymax)
    ax.set_yticks(range(0, 13, 3) if ymax == 13 else range(0, 17, 4))
    ax.set_xlabel("Forecast origin (checkups)")
    S.bold_ticks(ax)
axes[0].set_ylabel("Capacity MAPE (%)")
caps = S.subcaptions(fig, axes, [f"({'ab'[k]}) {p[1]}" for k, p in enumerate(PANELS)], gap_pt=8)
handles = [Patch(facecolor=S.MODEL[m]["c"], edgecolor=S.MODEL[m]["ec"], hatch=h, label=lab) for _, m, lab, h in ORDER]
leg = S.legend_below(fig, caps, handles, ncol=3, gap_pt=10, handlelength=1.6, columnspacing=1.4, handletextpad=0.45)
note = S.note_below(fig, [leg], "Error bars: 95 % cell-bootstrap CIs. Residual learner trained without target-dataset records.", gap_pt=6)
S.check_inside(fig, caps + [leg, note] + [a.xaxis.label for a in axes])
S.save(fig, "fig_external_transfer")
