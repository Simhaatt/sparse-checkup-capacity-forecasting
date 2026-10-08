"""Figure 4 - equal-cell capacity APE by future checkup step, origins 10/15/20 (104-cell cohort).

Data: data_used/fig4_error_by_horizon.csv (r4_hybrid_error_by_horizon.csv plus the number of cells
contributing to each step, counted from the saved test predictions). Values are plotted exactly as
measured: no smoothing, interpolation or synthetic points; no horizon-level CIs are drawn.
Run:  python fig_forecast_horizon.py
"""
import pandas as pd, matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.patheffects as pe
import jes_style as S

S.setup()
d = pd.read_csv(S.DATA / "fig4_error_by_horizon.csv")
SER = [("ARIMA_AICc", "ARIMA"), ("hybrid_sqrt_xgb", "Residual learner"), ("avg_ARIMA_hybrid", "ARIMA–residual average")]
LEG = {"Residual learner": "Residual learner (ours)", "ARIMA–residual average": "ARIMA–residual average (ours)"}

fig, axes = plt.subplots(1, 3, figsize=(S.WIDTH_MM * S.MM, 104 * S.MM), sharey=True,
                         gridspec_kw={"width_ratios": [1.25, 1.05, 0.9]})
fig.subplots_adjust(left=0.095, right=0.985, top=0.975, bottom=0.40, wspace=0.14)
for k, (ax, o) in enumerate(zip(axes, [10, 15, 20])):
    S.offset_spines(ax)
    g = d[d.origin.eq(o)].sort_values("step")
    for z, (col, name) in enumerate(SER):          # lines first ...
        st = S.MODEL[name]
        ax.plot(g.step, g[col], color=st["c"], lw=S.LW, zorder=2 + z, solid_capstyle="round")
    for z, (col, name) in enumerate(SER):          # ... then markers with a white ring, so crossings stay legible
        st = S.MODEL[name]
        ax.plot(g.step, g[col], ls="none", marker=st["m"], mfc=st["c"], mec=st["ec"], mew=0.9, ms=5.5, zorder=6 + z,
                path_effects=[pe.Stroke(linewidth=2.6, foreground="white"), pe.Normal()])
    step_ticks = [1, 5, 10, 15, 19] if o == 10 else ([1, 5, 10, 14] if o == 15 else [1, 3, 5, 7, 9])
    ax.set_xticks(step_ticks); ax.set_xlim(0.3, g.step.max() + 0.7)
    ax.set_ylim(0, 7.4); ax.set_yticks(range(0, 8))
    ax.set_xlabel("Steps ahead")
    S.bold_ticks(ax)
axes[0].set_ylabel("Capacity MAPE (%)")
caps = S.subcaptions(fig, axes, [f"({'abc'[k]}) Origin {o}" for k, o in enumerate([10, 15, 20])], gap_pt=8)
handles = [Line2D([], [], color=S.MODEL[n]["c"], marker=S.MODEL[n]["m"], mec=S.MODEL[n]["ec"], mew=1.2, ms=8, lw=S.LW, label=LEG.get(n, n)) for _, n in SER]
leg = S.legend_below(fig, caps, handles, ncol=3, gap_pt=10, handlelength=2.2, columnspacing=1.2, handletextpad=0.5)
# sample sizes come from the saved predictions (fig4_error_by_horizon.csv, n_cells column)
n_first = sorted(d[d.step.eq(1)].n_cells.unique()); n_last = sorted(d.sort_values("step").groupby("origin").tail(1).n_cells.unique())
assert len(n_first) == 1 and len(n_last) == 1, (n_first, n_last)
note = S.note_below(fig, [leg], f"Cells contributing: {n_first[0]} at step 1, falling to {n_last[0]} at the final step of each origin.", gap_pt=6)
S.check_inside(fig, caps + [leg, note] + [a.xaxis.label for a in axes])
S.save(fig, "fig_forecast_horizon")
