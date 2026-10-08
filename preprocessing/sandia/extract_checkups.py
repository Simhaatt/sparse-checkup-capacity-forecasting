"""Extract capacity-check (RPT) capacities from Sandia (Preger et al. 2020) cycle-level data.

Protocol (batteryarchive study summary): each round = capacity check (3 x 0-100% SOC cycles at 0.5C),
N cycles at the cell's condition, capacity check again. Check cycles are identified from cycle-level data as
  (a) cycles whose mean discharge voltage e_d/ah_d is clearly above the local rolling median
      (0.5C check discharge vs faster cycling discharge), and/or
  (b) for partial-SOC cells, cycles whose discharged Ah is clearly above the local rolling median (full vs partial).
Flagged cycles within 8 cycle indices of each other form one check block. Block capacity = median ah_d of its
flagged cycles. Check cycles must be full discharges (Ah above the cycling window + 0.15*nominal for partial-SOC cells,
> 0.5*nominal otherwise, and < 1.15*nominal); blocks with a single flagged cycle are dropped (except the initial check).
Cells cycled at 0-100% SOC with 0.5C discharge have check cycles identical to ageing cycles and
cannot be separated; they are excluded and listed.
Outputs: snl_checkups.csv (one row per cell x check), snl_checkup_report.txt
"""
import numpy as np
import pandas as pd

d = pd.read_csv("snl_cycle_data.csv").sort_values(["cell_id", "cycle_index"])
m = pd.read_csv("snl_cells_metadata.csv").set_index("cell_id")
d = d[d.ah_d.notna() & (d.ah_d > 0)].copy()
d["vd"] = d.e_d / d.ah_d

rows, excluded, report = [], [], []
for cid, g in d.groupby("cell_id", sort=True):
    md = m.loc[cid]
    g = g.reset_index(drop=True)
    full_dod = md.soc_min == 0 and md.soc_max == 100
    if full_dod and md.crate_d == 0.5:
        excluded.append((cid, "0-100% SOC at 0.5C discharge: check cycles indistinguishable from ageing cycles"))
        continue
    flag = np.zeros(len(g), bool)
    if md.crate_d > 0.5:
        med = g.vd.rolling(25, center=True, min_periods=5).median()
        res = g.vd - med
        mad = (res - res.median()).abs().median()
        flag |= (res > max(0.008, 6 * 1.4826 * mad)).to_numpy()
    if not full_dod:
        mah = g.ah_d.rolling(25, center=True, min_periods=5).median()
        flag |= (g.ah_d > 1.15 * mah).to_numpy()
    # a check cycle must be a full 0-100% discharge: above the cycling window and physically plausible
    win = (md.soc_max - md.soc_min) / 100.0
    lo = (min(win, 0.85) + 0.15) * md.ah if not full_dod else 0.5 * md.ah
    plaus = ((g.ah_d > lo) & (g.ah_d < 1.15 * md.ah)).to_numpy()
    if g.cycle_index[0] == 1:
        flag[:3] |= True  # initial check (cycles 1-3) per protocol
    flag &= plaus
    idx = np.flatnonzero(flag)
    if len(idx) == 0:
        excluded.append((cid, "no check cycles detected")); continue
    blocks, cur = [], [idx[0]]
    for i in idx[1:]:
        if g.cycle_index[i] - g.cycle_index[cur[-1]] <= 8:
            cur.append(i)
        else:
            blocks.append(cur); cur = [i]
    blocks.append(cur)
    efc = g.ah_d.cumsum() / md.ah  # equivalent full cycles discharged so far (inclusive)
    blocks = [b for j, b in enumerate(blocks) if len(b) >= 2 or (j == 0 and g.cycle_index[b[0]] == 1)]
    dropped_single = None
    for k, b in enumerate(blocks):
        b0 = b[0]
        rows.append(dict(cell_id=cid, checkup_idx=k, first_cycle=int(g.cycle_index[b0]), n_check_cycles=len(b),
                         t_days=g.test_time[b0] / 86400.0,
                         efc_before=float(efc[b0 - 1]) if b0 > 0 else 0.0,
                         Q=float(np.median(g.ah_d.to_numpy()[b])),
                         cathode=md.cathode, ah_nominal=md.ah, temperature=md.temperature,
                         soc_min=md.soc_min, soc_max=md.soc_max, crate_c=md.crate_c, crate_d=md.crate_d))
out = pd.DataFrame(rows)
out["Q0"] = out.groupby("cell_id").Q.transform("first")
out["SOH"] = out.Q / out.Q0
out.to_csv("snl_checkups.csv", index=False)

n = out.groupby("cell_id").size()
report.append(f"cells in metadata: {len(m)}; cells with checkups: {n.size}; excluded: {len(excluded)}")
for c, why in excluded:
    report.append(f"  excluded {c}: {why}")
report.append("checkups per cell:\n" + n.describe().round(1).to_string())
for o in (5, 8, 10, 12, 15, 20):
    report.append(f"cells with > {o} checkups (origin {o} leaves >=1 target): {(n > o).sum()}  | with >= {o}+5: {(n >= o + 5).sum()}")
report.append("by chemistry (cells, median checkups):\n" + out.groupby("cathode").cell_id.nunique().to_string()
              + "\n" + n.groupby(out.groupby('cell_id').cathode.first()).median().to_string())
last = out.groupby("cell_id").SOH.last()
report.append("final SOH quantiles:\n" + last.quantile([0, .25, .5, .75, 1]).round(3).to_string())
nonmono = out.groupby("cell_id").SOH.apply(lambda s: (s.diff() > 0.02).sum())
report.append(f"checkup-to-checkup capacity increases > 2% of Q0 (recovery/noise): total {int(nonmono.sum())} in {(nonmono > 0).sum()} cells")
open("snl_checkup_report.txt", "w").write("\n".join(report))
print("\n".join(report))
