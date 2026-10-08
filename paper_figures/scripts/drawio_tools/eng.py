"""Engineering-flowchart style for the JES diagrams (revision 5).

Rectangular blocks on a fixed grid, black/grey with a single blue accent, regular-weight body text with bold
headings, italic serif maths on the signal lines, thin 1 pt strokes and small arrowheads.
Canvas: 1000 px = 180 mm, so 1 px = 0.504 pt.
"""
from drawio_lib import Diagram

INK = "#1a1a1a"; GREY = "#6b6b6b"; LIGHT = "#f2f2f2"; MID = "#d9d9d9"; DARK = "#9e9e9e"
ACC = "#2a78d6"; ACC_F = "#e6f0fb"; ACC_D = "#1b5aa6"
HEAD, BODY, MATH, CAP = 21, 19, 22, 24          # 10.6 pt bold, 9.6 pt regular, 11 pt italic serif, 12 pt bold
SW, WIRE, HEADSZ = 2, 2, 7                       # 1 pt borders and wires
SERIF = "Times New Roman"


def block(d, x, y, w, h, head="", body="", fill="#ffffff", stroke=INK, sw=SW, dashed=False, head_fs=HEAD, body_fs=BODY):
    """Square-cornered block; bold heading line and optional regular body lines, centred."""
    d.box(x, y, w, h, "", fill=fill, stroke=stroke, sw=sw, rounded=0, check=False, dashed=dashed)
    nb = body.count("\n") + 1 if body else 0
    lh_h, lh_b = head_fs * 1.25, body_fs * 1.25
    tot = (lh_h if head else 0) + nb * lh_b
    y0 = y + (h - tot) / 2
    if head:
        d.text(x + 4, y0, w - 8, lh_h, head, fs=head_fs, align="center", color=INK, check=False)
    if body:
        d.text(x + 4, y0 + (lh_h if head else 0), w - 8, nb * lh_b, body, fs=body_fs, align="center", color=INK,
               bold=False, check=False)


def wire(d, pts, arrow=True, color=INK, dashed=False, sw=WIRE):
    return d.path(pts, color=color, sw=sw, arrow=arrow, dashed=dashed, dash="6 4", head=HEADSZ)


def math(d, x, y, w, text, align="center", color=INK, fs=MATH, h=28):
    d.text(x, y, w, h, text, fs=fs, align=align, color=color, bold=False, italic=True, font=SERIF, check=False)


def note(d, x, y, w, h, text, align="center", color=INK, fs=BODY, valign="middle"):
    d.text(x, y, w, h, text, fs=fs, align=align, color=color, bold=False, valign=valign, check=False)


def head(d, x, y, w, h, text, align="center", color=INK, fs=HEAD):
    d.text(x, y, w, h, text, fs=fs, align=align, color=color, check=False)


def sumnode(d, cx, cy, r=17, color=INK):
    d.box(cx - r, cy - r, 2 * r, 2 * r, "+", fs=24, fill="#ffffff", stroke=color, sw=SW, rounded=0, shape="ellipse",
          check=False, spacing=0, bold=False)


def gain(d, x, cy, w=44, h=40, text="0.5"):
    """Gain block: triangle pointing right with the gain value inside."""
    d.box(x, cy - h / 2, w, h, text, fs=17, fill="#ffffff", stroke=INK, sw=SW, rounded=0, shape="triangle",
          check=False, spacing=0, bold=False, align="left", extra="direction=east;spacingLeft=4;")


def junction(d, cx, cy, r=4, color=INK):
    d.dot(cx, cy, 2 * r, color, stroke=color, sw=1)


def subcap(d, x, w, y, text):
    d.text(x, y, w, 34, text, fs=CAP, align="center", color=INK, check=False)
