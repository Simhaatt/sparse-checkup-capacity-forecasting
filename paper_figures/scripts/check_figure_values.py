"""Assert every check value named in the figure brief against data_used/ (run before regenerating figures)."""
import pandas as pd
from pathlib import Path
D = Path(__file__).resolve().parents[1] / "data_used"
ok = []
def chk(name, got, want, tol=0.005):
    assert abs(got - want) <= tol, (name, got, want); ok.append((name, round(got, 4), want))
a = pd.read_csv(D / "fig3_primary_accuracy.csv").set_index(["method", "origin"]).MAPE
for (m, o), v in {("ARIMA_AICc", 10): 3.28, ("ARIMA_AICc", 15): 2.71, ("ARIMA_AICc", 20): 2.76, ("hybrid_sqrt_xgb", 10): 2.22, ("hybrid_sqrt_xgb", 15): 2.19,
                  ("hybrid_sqrt_xgb", 20): 2.18, ("avg_ARIMA_hybrid", 10): 2.35, ("avg_ARIMA_hybrid", 15): 2.14, ("avg_ARIMA_hybrid", 20): 2.07}.items():
    chk(f"Fig3 {m} o{o}", a[(m, o)], v)
assert ("r4_best_neural", 20) not in a.index; ok.append(("Fig3 re-tuned best neural at origin 20 absent (not run)", "-", "-"))
h = pd.read_csv(D / "fig4_error_by_horizon.csv").set_index(["origin", "step"])
chk("Fig4 o10 step1 ARIMA", h.loc[(10, 1), "ARIMA_AICc"], 0.21); chk("Fig4 o10 step1 residual", h.loc[(10, 1), "hybrid_sqrt_xgb"], 0.56)
chk("Fig4 o10 step19 ARIMA", h.loc[(10, 19), "ARIMA_AICc"], 6.53); chk("Fig4 o10 step19 residual", h.loc[(10, 19), "hybrid_sqrt_xgb"], 2.67)
chk("Fig4 o10 step19 average", h.loc[(10, 19), "avg_ARIMA_hybrid"], 3.78); chk("Fig4 o10 step19 cells", h.loc[(10, 19), "n_cells"], 81, 0)
t = pd.read_csv(D / "fig5_transfer.csv"); t = t[t.label.str.startswith("Residual")].set_index(["dataset", "origin"]).MAPE
for o, v in zip([3, 5, 8], [4.08, 3.61, 3.39]): chk(f"Fig5 cylindrical o{o}", t[("Kirkaldy 21700 (40 cells)", o)], v)
for o, v in zip([5, 10, 20], [4.40, 5.07, 1.87]): chk(f"Fig5 pouch o{o}", t[("Oxford pouch (8 cells)", o)], v)
f = pd.read_csv(D / "fig6_r4_decisions_luh_r3rule.csv").set_index(["method", "origin"]).F1
for o, (ar, hy) in {10: (0.87, 0.96), 15: (0.87, 0.94), 20: (0.90, 0.90)}.items():
    chk(f"Fig6d F1 ARIMA o{o}", f[("ARIMA_AICc", o)], ar); chk(f"Fig6d F1 residual o{o}", f[("hybrid_sqrt_xgb", o)], hy)
n = pd.read_csv(D / "fig6_sandia_cells_by_chemistry.csv").set_index(["origin", "chem"]).n_cells
chk("Fig6c NCA cells at origin 8", n[(8, "NCA")], 3, 0)
for r in ok: print("PASS", *r)
print(len(ok), "checks passed")
