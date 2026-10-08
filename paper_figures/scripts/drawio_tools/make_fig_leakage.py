"""Figure A.1 - resistance-alignment audit (editable draw.io), revision 5: engineering style, panels side by side.
Conceptual timing diagrams; no measured dates.

(a) Original join in the processed derivative: EIS ordinals 1..28 vs raw capacity row numbers (up to 5,998 in the
    audited cyclic cell); nearest-number join, no tolerance -> every row numbered above 28 gets sweep 28 (the last).
    Audit (01_PAPER_DETAILS.md 5.1): cyclic cell, origin 10: 9/10 resistance-prefix rows had future contributors
    (max ~295 d); origin 20: 19/20 (~85 d); calendar and profile cells: none.
(b) Corrected join: complete spectra, chronological backward-only, <= 14 d, missing kept missing. Outcome: one matched
    value per cell, all before origin 10 -> benchmark is capacity-only.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawio_lib import Diagram
from eng import *

W, H = 1000, 640
d = Diagram("fig_appendix_leakage_timeline", W, H)
d.box(0, 0, W, H, "", fill="#ffffff", stroke="none", sw=0, rounded=0, check=False, cid="canvas")
A0, A1, B0, B1 = 0, 482, 520, 988


def eis(x, y, label=None, fs=BODY):
    d.diamond(x, y, 16, INK, stroke=INK, sw=1)
    if label: note(d, x - 24, y + 8, 48, 24, label, fs=fs)


def cap(x, y, hollow=False):
    d.dot(x, y, 14, "#ffffff" if hollow else INK, stroke=INK, sw=SW)


def timing(x0, x1, ox, ye, yc, ya):
    note(d, x0, ye - 14, 84, 28, "EIS", align="right")
    note(d, x0, yc - 14, 84, 28, "Capacity", align="right")
    g0 = x0 + 96
    d.line(g0, ye, x1, ye, color=MID, sw=SW)
    d.line(g0, yc, x1, yc, color=MID, sw=SW)
    d.line(ox, ye - 30, ox, ya, color=INK, sw=SW, dashed=True, dash="8 4")
    note(d, ox - 80, ye - 58, 160, 26, "origin 10")
    wire(d, [(g0, ya), (x1, ya)])
    note(d, g0, ya + 2, x1 - g0, 24, "time (schematic)", align="right", color=GREY)
    return g0


# legend
LY = 14
eis(10, LY); note(d, 24, 0, 120, 28, "EIS sweep", align="left")
cap(150, LY); note(d, 164, 0, 140, 28, "capacity row", align="left")
cap(300, LY, hollow=True); note(d, 314, 0, 90, 28, "missing", align="left")
wire(d, [(410, LY), (462, LY)], dashed=True); note(d, 468, 0, 220, 28, "nearest-number match", align="left")
wire(d, [(700, LY), (752, LY)], color=ACC); note(d, 758, 0, 230, 28, "backward match", align="left")

# ============================ (a) original ============================
head(d, A0, 40, A1 - A0, 28, "Original join (processed derivative)")
KY = 146
note(d, A0, KY - 14, 84, 28, "Join key", align="right")
g = A0 + 96
d.line(g, KY, 218, KY, color=INK, sw=SW)
d.line(244, KY, A1, KY, color=INK, sw=SW)
d.line(222, KY + 9, 230, KY - 9, color=INK, sw=SW); d.line(232, KY + 9, 240, KY - 9, color=INK, sw=SW)
for x, lab in ((106, "1"), (130, "2"), (180, "27"), (206, "28")):
    eis(x, KY, lab)
note(d, 140, KY + 2, 30, 24, "…")
CX = [290, 340, 390, 440]
for x in CX: cap(x, KY)
note(d, 260, KY + 8, 210, 24, "rows ≫ 28")
BUS = KY - 38
for x in CX: d.path([(x, KY - 8), (x, BUS)], color=INK, sw=SW, dashed=True, dash="6 4")
wire(d, [(CX[-1], BUS), (206, BUS), (206, KY - 10)], dashed=True)
note(d, 196, BUS - 30, 286, 26, "nearest number: 28 (last sweep)", color=INK)

YE, YC, YA = 252, 322, 362
ox = A0 + 300
timing(A0, A1, ox, YE, YC, YA)
for x in (120, 190, 260, 360): eis(x, YE)
eis(462, YE, "28")
pre = [116, 146, 176, 206, 236, 266, 292]
for x in pre: cap(x, YC)
for x in (340, 390, 440): cap(x, YC)
for x in pre[::2]:
    wire(d, [(x + 5, YC - 7), (455, YE + 7)], dashed=True)

TY, RH2 = 404, 34
COLS = [("Origin", 92), ("Rows with future EIS", 230), ("Max. lead", 150)]
DATA = [("10", "9 / 10", "≈ 295 d"), ("20", "19 / 20", "≈ 85 d")]
x = A0 + 6
for ci, (hd, w) in enumerate(COLS):
    d.box(x, TY, w, RH2, hd, fs=HEAD - 1, fill=LIGHT, stroke=INK, sw=SW, rounded=0, check=False)
    for ri, row in enumerate(DATA):
        d.box(x, TY + (ri + 1) * RH2, w, RH2, row[ci], fs=BODY, fill="#ffffff", stroke=INK, sw=SW, rounded=0, bold=False, check=False)
    x += w
note(d, A0, TY + 3 * RH2 + 8, A1 - A0, 48, "Audited cyclic cell. Audited calendar and\nprofile cells: no future contributors.")

# ============================ (b) corrected ============================
head(d, B0, 40, B1 - B0, 28, "Corrected join (this study)")
RULES = [["Complete spectra only", "Backward in time only"], ["Within 14 days", "No match → missing"]]
rw = (B1 - B0) / 2
for r, row in enumerate(RULES):
    for c, txt in enumerate(row):
        d.box(B0 + c * rw, 88 + r * 40, rw, 40, txt, fs=BODY, fill=ACC_F, stroke=INK, sw=SW, rounded=0, bold=False, check=False)
oxb = B0 + 300
timing(B0, B1, oxb, YE, YC, YA)
SWEEPS = [B0 + 120, B0 + 228, B0 + 360, B0 + 452]
for x in SWEEPS: eis(x, YE)
CHK = [(B0 + 138, B0 + 120), (B0 + 184, None), (B0 + 248, B0 + 228), (B0 + 288, None), (B0 + 380, B0 + 360), (B0 + 468, B0 + 452)]
for x, m in CHK:
    cap(x, YC, hollow=m is None)
    if m is not None:
        wire(d, [(x - 3, YC - 8), (m + 3, YE + 9)], color=ACC)
# 14-day window on one checkup
d.line(B0 + 248 - 40, YE - 20, B0 + 248, YE - 20, color=ACC_D, sw=SW)
d.line(B0 + 208, YE - 26, B0 + 208, YE - 14, color=ACC_D, sw=SW); d.line(B0 + 248, YE - 26, B0 + 248, YE - 14, color=ACC_D, sw=SW)
note(d, B0 + 150, YE - 52, 110, 24, "≤ 14 d", color=ACC_D, align="center")
# forward match refused
wire(d, [(B0 + 188, YC - 8), (B0 + 222, YE + 10)], color=GREY, dashed=True)
note(d, B0 + 178, YE + 14, 26, 26, "✕", color=INK)

d.box(B0, TY, B1 - B0, 3 * RH2, "", fill=LIGHT, stroke=INK, sw=SW, rounded=0, check=False)
head(d, B0 + 12, TY + 6, B1 - B0 - 24, 26, "Outcome", align="left")
note(d, B0 + 12, TY + 34, B1 - B0 - 24, 64, "One matched value per cell, all before origin 10,\nso the benchmark uses capacity only.", align="left")

SY = TY + 3 * RH2 + 70
subcap(d, A0, A1 - A0, SY, "(a) Faulty join by row number")
subcap(d, B0, B1 - B0, SY, "(b) Corrected backward join")
Hf = SY + 40
d.cells[0] = d.cells[0].replace(f'height="{H}"', f'height="{Hf}"')
d.save(Path(__file__).resolve().parents[2] / "editable" / "fig_appendix_leakage_timeline.drawio")
print("canvas", W, Hf)
