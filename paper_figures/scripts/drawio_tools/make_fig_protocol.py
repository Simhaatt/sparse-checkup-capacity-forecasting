"""Figure 1 - forecasting evaluation protocol (editable draw.io), revision 5: engineering style, panels side by side.

(a) 5-fold cell-grouped matrix (test group on the diagonal), zoom of one fold's development cells into a proportional
    role bar (fitting 135-136 | validation 27 | calibration 20), and a role-use table. 01_PAPER_DETAILS.md 4.1.
(b) Information available at the forecast origin, as a timeline table: rows are information sources, columns are
    checkups up to the origin and after it.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawio_lib import Diagram
from eng import *

W, H = 1000, 600
d = Diagram("fig_forecast_protocol", W, H)
d.box(0, 0, W, H, "", fill="#ffffff", stroke="none", sw=0, rounded=0, check=False, cid="canvas")
A0, A1, B0, B1 = 0, 482, 520, 988

# ============================ (a) ============================
head(d, A0, 0, A1 - A0, 28, "228 cells in 5 groups (by cell ID)")
MX, CW, RH, RG, Y0 = 92, 78, 32, 6, 62
for j in range(5):
    note(d, MX + j * CW, 34, CW, 24, f"G{j + 1}", color=GREY)
for k in range(5):
    y = Y0 + k * (RH + RG)
    note(d, A0, y, MX - 10, RH, f"Fold {k + 1}", align="right")
    for j in range(5):
        if j == k:
            d.box(MX + j * CW + 2, y, CW - 4, RH, "Test", fs=BODY, fill=ACC, stroke=ACC_D, sw=SW, rounded=0, color="#ffffff", check=False)
        else:
            d.box(MX + j * CW + 2, y, CW - 4, RH, "", fill="#ffffff", stroke=DARK, sw=SW, rounded=0, check=False)
y_end = Y0 + 5 * (RH + RG) - RG
BY, BH, BW = y_end + 40, 36, 5 * CW
tot = 135.5 + 27 + 20
wf = round(BW * 135.5 / tot); wv = round(BW * 27 / tot); wc = BW - wf - wv
wire(d, [(MX + 2, y_end + 2), (MX, BY - 2)], arrow=False, color=GREY, dashed=True)
wire(d, [(MX + 4 * CW - 2, y_end + 2), (MX + BW, BY - 2)], arrow=False, color=GREY, dashed=True)
d.box(MX, BY, wf, BH, "Fitting  135–136", fs=BODY, fill="#ffffff", stroke=INK, sw=SW, rounded=0, bold=False, check=False)
d.box(MX + wf, BY, wv, BH, "27", fs=BODY, fill=MID, stroke=INK, sw=SW, rounded=0, bold=False, check=False)
d.box(MX + wf + wv, BY, wc, BH, "20", fs=BODY, fill=DARK, stroke=INK, sw=SW, rounded=0, bold=False, check=False)
note(d, A0, BY, MX - 10, BH, "Cells", align="right")
note(d, MX + wf - 14, BY + BH + 2, wv + 28, 24, "Val.", color=GREY)
note(d, MX + wf + wv - 8, BY + BH + 2, wc + 16, 24, "Cal.", color=GREY)
# role-use table
TY, RH2 = BY + BH + 44, 34
COLS = [("Model", 110), ("Tune", 104), ("Fit", 140), ("Calibrate", 122)]
ROWS = [("Point", "Val.", "Fit + Cal.", "–"), ("Interval", "–", "Fit", "Cal.")]
x = A0 + 4
for ci, (hd, w) in enumerate(COLS):
    d.box(x, TY, w, RH2, hd, fs=HEAD - 1, fill=LIGHT, stroke=INK, sw=SW, rounded=0, check=False)
    for ri, row in enumerate(ROWS):
        d.box(x, TY + (ri + 1) * RH2, w, RH2, row[ci], fs=BODY, fill="#ffffff", stroke=INK, sw=SW, rounded=0, bold=(ci == 0), check=False)
    x += w
note(d, A0, TY + 3 * RH2 + 10, A1 - A0, 48, "Scored: 104 cells with > 20 checkups,\neach tested once.", color=INK)
A_END = TY + 3 * RH2 + 58

# ============================ (b) ============================
head(d, B0, 0, B1 - B0, 28, "Information at forecast origin o")
LX, C0, OX, C1 = B0, B0 + 128, B0 + 290, B1
AXY = 52
wire(d, [(C0, AXY), (C1, AXY)], arrow=True)
for k, x in enumerate(range(C0 + 14, C1 - 16, 27)):
    d.line(x, AXY - 6, x, AXY + 6, color=INK, sw=SW)
note(d, C0, AXY + 6, OX - C0, 26, "checkups 1 … o", color=INK)
note(d, OX, AXY + 6, C1 - OX, 26, "o + 1, o + 2, …", color=INK)
RY0, RH3 = AXY + 44, 50
ROWS = [("Capacity", ("input", "acc"), ("target (scored)", "white")),
        ("Descriptors", ("input", "acc"), ("input (fixed)", "acc")),
        ("Schedule", ("input", "acc"), ("input (planned)", "acc")),
        ("Resistance", ("not used", "white"), ("excluded", "grey")),
        ("Throughput", ("not used", "white"), ("excluded", "grey"))]
FILL = {"acc": ACC_F, "white": "#ffffff", "grey": MID, "acc-span": ACC_F}
for i, (lab, pre, post) in enumerate(ROWS):
    y = RY0 + i * RH3
    note(d, LX, y, C0 - LX - 10, RH3, lab, align="right")
    if pre[1] == "acc-span":
        d.box(C0, y, C1 - C0, RH3, pre[0], fs=BODY, fill=FILL["acc"], stroke=INK, sw=SW, rounded=0, bold=False, check=False)
        continue
    d.box(C0, y, OX - C0, RH3, pre[0], fs=BODY, fill=FILL[pre[1]], stroke=INK, sw=SW, rounded=0, bold=False,
          color=GREY if pre[0] == "not used" else INK, check=False)
    d.box(OX, y, C1 - OX, RH3, post[0], fs=BODY, fill=FILL[post[1]], stroke=INK, sw=SW, rounded=0, bold=False, check=False)
y_tab = RY0 + len(ROWS) * RH3
d.line(OX, AXY - 22, OX, y_tab + 4, color=ACC_D, sw=3, dashed=True, dash="7 5")
math(d, OX + 6, AXY - 34, 40, "o", align="left", color=ACC_D)
note(d, B0, y_tab + 16, B1 - B0, 48, "o = 10, 15 or 20 measured checkups.\nBlue: model input; grey: excluded.", color=INK)
B_END = y_tab + 64

SY = max(A_END, B_END) + 14
subcap(d, A0, A1 - A0, SY, "(a) Cell-grouped evaluation")
subcap(d, B0, B1 - B0, SY, "(b) Information boundary")
Hf = SY + 40
d.cells[0] = d.cells[0].replace(f'height="{H}"', f'height="{Hf}"')
d.save(Path(__file__).resolve().parents[2] / "editable" / "fig_forecast_protocol.drawio")
print("canvas", W, Hf)
