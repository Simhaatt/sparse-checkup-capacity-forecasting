"""Figure 2 - prior-anchored residual forecasting (editable draw.io), revision 5: engineering block diagram.

Signals: target-cell prefix -> square-root prior p(i) (OLS, b <= 0) and prefix features z(i); XGBoost residual
learner (trained offline on other cells at simulated origins, target SOH - p) -> r_hat(i); y1 = p + r_hat (residual
learner); y2 = ARIMA (AICc) per cell; y = 0.5 y1 + 0.5 y2. Facts: 01_PAPER_DETAILS.md section 6; r4_hybrid.py.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from drawio_lib import Diagram
from eng import *

W, H = 1000, 446
d = Diagram("fig_residual_method", W, H)
d.box(0, 0, W, H, "", fill="#ffffff", stroke="none", sw=0, rounded=0, check=False, cid="canvas")

# rows (centre y) and block geometry
YX, YP, YA = 200, 300, 400            # XGBoost, prior, ARIMA rows
BH = 70

# offline training (other cells only)
d.box(300, 2, 420, 124, "", fill="#ffffff", stroke=GREY, sw=SW, rounded=0, dashed=True, check=False)
note(d, 310, 6, 400, 24, "Offline training on other cells", align="left", color=GREY)
block(d, 312, 38, 168, 74, "Other cells", "simulated origins")
block(d, 532, 38, 168, 74, "Fit", "target SOH − p")
wire(d, [(480, 75), (532, 75)])

# input
block(d, 0, 150, 140, 300, "Target cell", "measured\nprefix\nSOH(1…o)", fill=LIGHT)

# component blocks
block(d, 532, YX - BH / 2, 168, BH, "XGBoost", "residual learner", fill=ACC_F, stroke=ACC, sw=2.5)
block(d, 204, YP - BH / 2, 206, BH, "Square-root prior", "OLS fit, b ≤ 0")
block(d, 204, YA - BH / 2, 206, BH, "ARIMA", "AICc, per cell")
wire(d, [(616, 112), (616, YX - BH / 2)])
math(d, 622, 126, 40, "θ", align="left")

# input -> blocks
wire(d, [(140, YX - 15), (532, YX - 15)]);  math(d, 160, YX - 46, 120, "z(i)", align="left")
wire(d, [(140, YP), (204, YP)])
wire(d, [(140, YA), (204, YA)])
# prior -> summation and -> residual learner (as a feature)
S1X, S1Y = 770, 250
junction(d, 450, YP)
wire(d, [(410, YP), (S1X, YP), (S1X, S1Y + 17)]);  math(d, 560, YP - 30, 80, "p(i)")
wire(d, [(450, YP), (450, YX + 15), (532, YX + 15)], arrow=True)
wire(d, [(700, YX), (S1X, YX), (S1X, S1Y - 17)], color=ACC);  math(d, 700, YX - 32, 70, "r̂(i)", color=ACC_D)
sumnode(d, S1X, S1Y, color=ACC)
# gains and final summation
GX = 812
S2X, S2Y = 912, 325
wire(d, [(S1X + 17, S1Y), (GX, S1Y)], color=ACC)
math(d, GX - 26, S1Y - 48, 90, "ŷ₁(i)", color=ACC_D)
gain(d, GX, S1Y)
wire(d, [(410, YA), (GX, YA)]);  math(d, 560, YA - 30, 80, "ŷ₂(i)")
gain(d, GX, YA)
wire(d, [(GX + 44, S1Y), (S2X, S1Y), (S2X, S2Y - 17)])
wire(d, [(GX + 44, YA), (S2X, YA), (S2X, S2Y + 17)])
sumnode(d, S2X, S2Y)
wire(d, [(S2X + 17, S2Y), (990, S2Y)]);  math(d, 932, S2Y - 32, 58, "ŷ(i)")
d.save(Path(__file__).resolve().parents[2] / "editable" / "fig_residual_method.drawio")
