"""Tiny helper to write genuine, editable draw.io (mxGraphModel) files.

Every element is a native draw.io vertex or edge (rectangles, ellipses, text, connectors),
with plain (non-HTML) labels so the same file renders identically in draw.io and in the
headless mxGraph renderer used for the PDF/SVG/PNG exports. Line breaks are explicit.
A simple width check guards against text overflowing its box.
"""
from xml.sax.saxutils import escape

FONT = "Arial"
INK = "#0b0b0b"; INK2 = "#52514e"
CHAR_W = 0.60   # conservative average glyph width of Arial Bold, in units of font size


def _attr(s):
    return escape(s, {'"': "&quot;", "\n": "&#xa;"})


class Diagram:
    def __init__(self, name, width, height):
        self.name, self.W, self.H = name, width, height
        self.cells = []; self.n = 2; self.warnings = []; self.dy = 0  # dy: vertical offset applied to every new element

    def _id(self, cid=None):
        if cid: return cid
        self.n += 1; return f"c{self.n}"

    def _check(self, text, w, fs, pad=16):
        for line in text.split("\n"):
            need = len(line) * CHAR_W * fs + pad
            if need > w + 0.5:
                self.warnings.append(f"[{self.name}] text may overflow ({need:.0f}>{w}): {line!r}")

    def box(self, x, y, w, h, text="", fs=22, fill="#ffffff", stroke=INK, sw=2.5, rounded=1, bold=True, color=INK,
            align="center", valign="middle", dashed=False, cid=None, shape=None, extra="", check=True, spacing=8, italic=False, font=None):
        style = (f"{'shape=' + shape + ';' if shape else ''}rounded={rounded};arcSize=8;whiteSpace=nowrap;html=0;"
                 f"fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontFamily={font or FONT};fontSize={fs};"
                 f"fontStyle={(1 if bold else 0) + (2 if italic else 0)};fontColor={color};align={align};verticalAlign={valign};"
                 f"spacing={spacing};{'dashed=1;dashPattern=8 5;' if dashed else ''}{extra}")
        if check and text: self._check(text, w, fs)
        y = y + self.dy
        cid = self._id(cid)
        self.cells.append(f'<mxCell id="{cid}" value="{_attr(text)}" style="{style}" vertex="1" parent="1">'
                          f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')
        return cid

    def titled_box(self, x, y, w, h, title, body, fs_title=25, fs_body=22, fill="#ffffff", stroke=INK, sw=2.5, pad=12, title_h=32):
        """A box with a larger bold title line and a smaller body, as separate editable objects."""
        bid = self.box(x, y, w, h, "", fill=fill, stroke=stroke, sw=sw, check=False)  # box/text apply self.dy themselves
        self.text(x + pad, y + 8, w - 2 * pad, title_h, title, fs=fs_title)
        n = body.count("\n") + 1
        self.text(x + pad, y + 8 + title_h, w - 2 * pad, h - title_h - 14, body, fs=fs_body, valign="top")
        return bid

    def text(self, x, y, w, h, text, fs=22, color=INK, align="left", valign="middle", bold=True, cid=None, check=True, italic=False, font=None):
        return self.box(x, y, w, h, text, fs=fs, fill="none", stroke="none", sw=0, rounded=0, bold=bold, color=color,
                        align=align, valign=valign, cid=cid, check=check, spacing=2, italic=italic, font=font)

    def ellipse(self, x, y, d, fill, stroke=INK, sw=2.5, cid=None):
        return self.box(x, y, d, d, "", fill=fill, stroke=stroke, sw=sw, rounded=0, shape="ellipse", cid=cid, check=False)

    def line(self, x1, y1, x2, y2, color=INK, sw=2.5, dashed=False, arrow=False, start_arrow=False, cid=None, dash="10 6"):
        cid = self._id(cid); y1 += self.dy; y2 += self.dy
        style = (f"endArrow={'block' if arrow else 'none'};endFill=1;endSize=10;startArrow={'block' if start_arrow else 'none'};startFill=1;startSize=10;"
                 f"html=0;strokeColor={color};strokeWidth={sw};{'dashed=1;dashPattern=' + dash + ';' if dashed else ''}")
        self.cells.append(f'<mxCell id="{cid}" value="" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')
        return cid

    def edge(self, src, tgt, points=(), color=INK, sw=3, dashed=False, label="", fs=20, exit=None, entry=None, cid=None, arrow=True, straight=False):
        cid = self._id(cid)
        def side(p, pre):
            return "" if p is None else f"{pre}X={p[0]};{pre}Y={p[1]};{pre}Dx=0;{pre}Dy=0;"
        style = (f"{'' if straight else 'edgeStyle=orthogonalEdgeStyle;'}rounded=0;orthogonalLoop=1;html=0;endArrow={'block' if arrow else 'none'};endFill=1;endSize=10;"
                 f"strokeColor={color};strokeWidth={sw};fontFamily={FONT};fontSize={fs};fontStyle=1;fontColor={INK};labelBackgroundColor=#ffffff;"
                 f"{'dashed=1;dashPattern=8 5;' if dashed else ''}{side(exit, 'exit')}{side(entry, 'entry')}")
        pts = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in points)
        arr = f'<Array as="points">{pts}</Array>' if points else ""
        self.cells.append(f'<mxCell id="{cid}" value="{_attr(label)}" style="{style}" edge="1" parent="1" source="{src}" target="{tgt}">'
                          f'<mxGeometry relative="1" as="geometry">{arr}</mxGeometry></mxCell>')
        return cid

    def path(self, pts, color=INK, sw=2.5, dashed=False, arrow=False, curved=False, dash="10 6", cid=None, label="", fs=21, head=10):
        """Free polyline / smooth curve through absolute points (native draw.io edge with waypoints)."""
        cid = self._id(cid); pts = [(x, y + self.dy) for x, y in pts]
        style = (f"{'curved=1;' if curved else ''}rounded=0;html=0;endArrow={'block' if arrow else 'none'};endFill=1;endSize={head};startArrow=none;"
                 f"strokeColor={color};strokeWidth={sw};fontFamily={FONT};fontSize={fs};fontStyle=1;fontColor={INK};labelBackgroundColor=#ffffff;"
                 f"{'dashed=1;dashPattern=' + dash + ';' if dashed else ''}")
        mid = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
        arr = f'<Array as="points">{mid}</Array>' if mid else ""
        self.cells.append(f'<mxCell id="{cid}" value="{_attr(label)}" style="{style}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                          f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/><mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>{arr}</mxGeometry></mxCell>')
        return cid

    def dot(self, cx, cy, d, fill, stroke=INK, sw=2.5):
        return self.ellipse(cx - d / 2, cy - d / 2, d, fill, stroke=stroke, sw=sw)

    def diamond(self, cx, cy, d, fill, stroke=INK, sw=2.5):
        return self.box(cx - d / 2, cy - d / 2, d, d, "", fill=fill, stroke=stroke, sw=sw, rounded=0, shape="rhombus", check=False)

    def node(self, cx, cy, d, text, fill, stroke=INK, sw=3, fs=30, cid=None):
        """Circular operator node (e.g. a summation '+'), editable as an ellipse with a label."""
        return self.box(cx - d / 2, cy - d / 2, d, d, text, fs=fs, fill=fill, stroke=stroke, sw=sw, rounded=0, shape="ellipse", check=False, spacing=0, cid=cid)

    def xml(self):
        body = "".join(self.cells)
        return (f'<mxfile host="drawio" version="24.7.17"><diagram id="{self.name}" name="{self.name}">'
                f'<mxGraphModel dx="{self.W}" dy="{self.H}" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" '
                f'page="1" pageScale="1" pageWidth="{self.W}" pageHeight="{self.H}" background="#ffffff" math="0" shadow="0">'
                f'<root><mxCell id="0"/><mxCell id="1" parent="0"/>{body}</root></mxGraphModel></diagram></mxfile>')

    def subcaption(self, y, text, fs=28):
        """Centred '(a) ...' sub-caption placed BELOW its panel."""
        return self.text(0, y, self.W, 40, text, fs=fs, align="center")

    def save(self, path):
        open(path, "w", encoding="utf-8").write(self.xml())
        for w in self.warnings: print("WARN", w)
        return path


# ---------------------------------------------------------------- modern-figure glyphs (all native shapes)
def card(d, x, y, w, h, title="", num=None, accent=INK, fill="#f6f6f4", stroke="#cfcdc6", fs=22, head_h=64, dashed=False):
    """Rounded group card with an optional numbered badge and a (possibly two-line) bold title."""
    d.box(x, y, w, h, "", fill=fill, stroke=stroke, sw=2, rounded=1, check=False, dashed=dashed, extra="arcSize=4;")
    tx = x + 12
    if num is not None:
        d.node(x + 12 + 17, y + head_h / 2, 34, str(num), fill=accent, stroke=accent, sw=2, fs=21)
        d.cells[-1] = d.cells[-1].replace(f"fontColor={INK}", "fontColor=#ffffff")
        tx = x + 56
    if title:
        d.text(tx, y + 4, x + w - tx - 8, head_h - 8, title, fs=fs)


def axes(d, x, y, w, h, color="#52514e", sw=2):
    """L-shaped mini-plot axes with small arrowheads (top-left corner at x, y)."""
    d.path([(x, y + h), (x, y)], color=color, sw=sw, arrow=True)
    d.path([(x, y + h), (x + w, y + h)], color=color, sw=sw, arrow=True)


def battery(d, x, y, w, h, level=0.7, color="#104281"):
    d.box(x, y, w, h, "", fill="#ffffff", stroke=color, sw=2.5, rounded=1, check=False)
    d.box(x + w, y + h * 0.3, w * 0.08, h * 0.4, "", fill=color, stroke=color, sw=1, rounded=0, check=False)
    d.box(x + 4, y + 4, (w - 8) * level, h - 8, "", fill=color, stroke="none", sw=0, rounded=0, check=False)


def tree(d, x, y, w, h, color="#2a78d6"):
    """Small decision-tree glyph: root, two children, four leaves."""
    r = 8
    pts = {"r": (x + w / 2, y + r), "a": (x + w / 4, y + h / 2), "b": (x + 3 * w / 4, y + h / 2)}
    leaves = [(x + w / 8 * k, y + h - r) for k in (1, 3, 5, 7)]
    for (px, py) in (pts["a"], pts["b"]):
        d.line(pts["r"][0], pts["r"][1], px, py, color=color, sw=2.5)
    for i, (lx, ly) in enumerate(leaves):
        p = pts["a"] if i < 2 else pts["b"]
        d.line(p[0], p[1], lx, ly, color=color, sw=2.5)
    for (px, py) in [pts["r"], pts["a"], pts["b"]]:
        d.dot(px, py, 2 * r, color, stroke=color, sw=1.5)
    for (lx, ly) in leaves:
        d.dot(lx, ly, 2 * r - 2, "#ffffff", stroke=color, sw=2.5)


def stack(d, x, y, w, h, n=3, off=8, stroke="#52514e", fill="#ffffff", curve_color="#898781"):
    """Stack of n offset cards, each with a tiny fade curve (a set of other cells)."""
    import math
    for k in range(n - 1, -1, -1):
        cx, cy = x + k * off, y - k * off
        d.box(cx, cy, w, h, "", fill=fill, stroke=stroke, sw=2, rounded=0, check=False)
    pts = [(x + 8 + t * (w - 16) / 5, y + 10 + (h - 20) * math.sqrt(t / 5)) for t in range(6)]
    d.path(pts, color=curve_color, sw=2.5, curved=True)
