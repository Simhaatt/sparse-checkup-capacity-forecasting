"""Render .drawio files to SVG, vector PDF and 600-dpi PNG with the mxGraph engine (the library draw.io is built on).

Usage: python render_drawio.py file.drawio [...] --outroot ../   (writes svg/, figures/, png_600dpi/)
The SVG is produced by mxGraph's own SVG canvas (plain <text>, no HTML), then printed to PDF by
headless Chromium at the final journal width (180 mm), so the PDF stays vector.
"""
import argparse, re, html, json
from pathlib import Path
from playwright.sync_api import sync_playwright

MX = Path(__file__).resolve().parent / "package" / "javascript"
WIDTH_MM = 180.0

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<script>var mxBasePath='%s/src'; var mxLoadResources=false; var mxLoadStylesheets=false; var mxForceIncludes=false;</script>
<script src="%s/mxClient.min.js"></script></head><body style="margin:0;background:#fff"><div id="g" style="position:absolute;left:0;top:0;width:4000px;height:4000px"></div>
<script>
function renderModel(xmlText){
  var doc = mxUtils.parseXml(xmlText);
  var container = document.getElementById('g');
  var graph = new mxGraph(container); graph.setHtmlLabels(false);
  var codec = new mxCodec(doc); codec.decode(doc.documentElement, graph.getModel());
  var pad = 6;
  var b = graph.getGraphBounds();
  var svgDoc = mxUtils.createXmlDocument();
  var root = (svgDoc.createElementNS != null) ? svgDoc.createElementNS(mxConstants.NS_SVG, 'svg') : svgDoc.createElement('svg');
  root.setAttribute('xmlns', mxConstants.NS_SVG); root.setAttribute('xmlns:xlink', mxConstants.NS_XLINK);
  var W = Math.ceil(b.width + 2*pad), H = Math.ceil(b.height + 2*pad);
  root.setAttribute('width', W); root.setAttribute('height', H); root.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
  root.setAttribute('version', '1.1');
  var bg = svgDoc.createElementNS(mxConstants.NS_SVG, 'rect'); bg.setAttribute('x',0); bg.setAttribute('y',0); bg.setAttribute('width',W); bg.setAttribute('height',H); bg.setAttribute('fill','#ffffff'); root.appendChild(bg);
  var group = svgDoc.createElementNS(mxConstants.NS_SVG, 'g'); root.appendChild(group);
  var canvas = new mxSvgCanvas2D(group); canvas.translate(Math.floor(pad - b.x), Math.floor(pad - b.y)); canvas.scale(1);
  canvas.textEnabled = true; canvas.foEnabled = false;
  var exp = new mxImageExport(); exp.includeOverlays = false;
  exp.drawState(graph.getView().getState(graph.model.root), canvas);
  return {svg: mxUtils.getXml(root), w: W, h: H, bx: b.x, by: b.y, bw: b.width};
}
</script></body></html>"""


def model_xml(path):
    s = Path(path).read_text(encoding="utf-8")
    m = re.search(r"<mxGraphModel.*?</mxGraphModel>", s, re.S)
    return m.group(0)


def render(files, outroot):
    outroot = Path(outroot)
    for d in ("svg", "figures", "png_600dpi"): (outroot / d).mkdir(parents=True, exist_ok=True)
    page_html = PAGE % (MX.as_posix(), MX.as_posix())
    tmp = Path(__file__).resolve().parent / "_mx_page.html"; tmp.write_text(page_html, encoding="utf-8")
    with sync_playwright() as p:
        br = p.chromium.launch()
        pg = br.new_page(); pg.goto(tmp.as_uri())
        for f in files:
            f = Path(f); res = pg.evaluate("x => renderModel(x)", model_xml(f))
            W, H = res["w"], res["h"]; h_mm = WIDTH_MM * H / W
            svg = res["svg"].replace(f'width="{W}" height="{H}"', f'width="{WIDTH_MM}mm" height="{h_mm:.2f}mm"', 1)
            svg = '<?xml version="1.0" encoding="UTF-8"?>\n' + svg
            (outroot / "svg" / f"{f.stem}.svg").write_text(svg, encoding="utf-8")
            doc = f'<!doctype html><html><head><meta charset="utf-8"><style>@page{{size:{WIDTH_MM}mm {h_mm:.2f}mm;margin:0}}html,body{{margin:0;padding:0;background:#fff}}svg{{display:block;width:{WIDTH_MM}mm;height:{h_mm:.2f}mm}}</style></head><body>{res["svg"]}</body></html>'
            p2 = br.new_page(); p2.set_content(doc)
            p2.pdf(path=str(outroot / "figures" / f"{f.stem}.pdf"), width=f"{WIDTH_MM}mm", height=f"{h_mm:.2f}mm", print_background=True, margin={"top": "0", "bottom": "0", "left": "0", "right": "0"})
            p2.close()
            px_w = round(WIDTH_MM / 25.4 * 600); scale = px_w / W
            p3 = br.new_page(viewport={"width": W, "height": H}, device_scale_factor=scale)
            p3.set_content(f'<html><body style="margin:0;background:#fff">{res["svg"]}</body></html>')
            p3.screenshot(path=str(outroot / "png_600dpi" / f"{f.stem}.png"), clip={"x": 0, "y": 0, "width": W, "height": H})
            p3.close()
            print("bounds x", res["bx"], "w", res["bw"]); print(f"rendered {f.name}: canvas {W}x{H}px -> {WIDTH_MM}x{h_mm:.1f} mm, 1 px = {WIDTH_MM / W * 72 / 25.4:.3f} pt")
        br.close()
    tmp.unlink()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("files", nargs="+"); ap.add_argument("--outroot", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args(); render(a.files, a.outroot)
