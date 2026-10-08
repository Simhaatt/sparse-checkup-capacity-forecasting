"""Contact sheet of all seven figures (rendered from the 600-dpi PNGs)."""
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
R = Path(__file__).resolve().parents[2]
names = ["fig_forecast_protocol", "fig_residual_method", "fig_primary_accuracy", "fig_forecast_horizon",
         "fig_external_transfer", "fig_interval_decisions", "fig_appendix_leakage_timeline"]
labels = ["Fig. 1", "Fig. 2", "Fig. 3", "Fig. 4", "Fig. 5", "Fig. 6", "Fig. A.1"]
W = 1400; tiles = []
font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", 34)
for n, l in zip(names, labels):
    im = Image.open(R / "png_600dpi" / f"{n}.png").convert("RGB"); im.thumbnail((W, 4000))
    t = Image.new("RGB", (W + 40, im.height + 90), "white"); d = ImageDraw.Draw(t)
    d.text((20, 18), f"{l}  {n}.pdf", fill="#0b0b0b", font=font); t.paste(im, (20, 70)); d.rectangle([10, 60, W + 30, im.height + 80], outline="#c3c2b7", width=3)
    tiles.append(t)
cols = 2; rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
H = sum(max(t.height for t in r) for r in rows); sheet = Image.new("RGB", (cols * (W + 40), H), "white"); y = 0
for r in rows:
    for j, t in enumerate(r): sheet.paste(t, (j * (W + 40), y))
    y += max(t.height for t in r)
sheet.save(R / "QA_contact_sheet.png"); print(sheet.size)
