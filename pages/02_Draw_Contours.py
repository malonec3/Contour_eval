import re
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon

# --------------------------- page --------------------------------------------
st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")

st.markdown(
    "<style>div[data-testid='column']{padding-left:.25rem;padding-right:.25rem}</style>",
    unsafe_allow_html=True,
)

# --------------------------- canvas & grid settings ---------------------------
CANVAS_W = 380
CANVAS_H = 380
MM_SPAN   = 20.0              # world is [-10, +10] mm in both axes
PX_PER_MM = CANVAS_W / MM_SPAN

GRID      = (256, 256)        # raster grid for masks/metrics
RESAMPLE_N = 400              # perimeter resampling count

# Colors used for drawing A/B
A_FILL,  A_STROKE = "rgba(0, 0, 255, 0.20)", "blue"
B_FILL,  B_STROKE = "rgba(255, 0, 0, 0.20)", "red"

# --------------------------- state -------------------------------------------
st.session_state.setdefault("polys_A", [])          # list of Fabric polygon dicts (tagged A)
st.session_state.setdefault("polys_B", [])          # list of Fabric polygon dicts (tagged B)
st.session_state.setdefault("canvas_seed", 0)       # bump to force remount only on Reset
st.session_state.setdefault("draw_results", None)   # persisted plots

# --------------------------- helpers -----------------------------------------
def grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1):
    """Fabric objects for grid + 10mm scale bar (non-selectable, non-evented)."""
    objs = []
    step_minor = PX_PER_MM * minor
    step_major = PX_PER_MM * major

    def add_line(x1,y1,x2,y2, color):
        objs.append({
            "type":"line","x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
            "stroke":color,"strokeWidth":1,"selectable":False,"evented":False,
            "excludeFromExport":True
        })

    # minor grid
    x = 0.0
    while x <= width + 0.5:
        add_line(x, 0, x, height, "#ebebeb"); x += step_minor
    y = 0.0
    while y <= height + 0.5:
        add_line(0, y, width, y, "#ebebeb"); y += step_minor

    # major grid
    x = 0.0
    while x <= width + 0.5:
        add_line(x, 0, x, height, "#d2d2d2"); x += step_major
    y = 0.0
    while y <= height + 0.5:
        add_line(0, y, width, y, "#d2d2d2"); y += step_major

    # border
    objs.append({
        "type":"rect","left":0,"top":0,"width":float(width),"height":float(height),
        "fill":"","stroke":"#c8c8c8","strokeWidth":1,"selectable":False,"evented":False,
        "excludeFromExport":True
    })

    # 10 mm scale bar (bottom-left)
    bar_px = 10 * PX_PER_MM; margin = 10.0; y0 = height - margin; x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0,"y1":y0-5,"x2":x0,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0+bar_px,"y1":y0-5,"x2":x0+bar_px,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
    ]
    return objs

GRID_OBJS = grid_objects()

def build_initial_json():
    """Grid + whatever polygons we’ve already tagged."""
    return {"objects": [*GRID_OBJS, *st.session_state.polys_A, *st.session_state.polys_B]}

def _fabric_polygon_points(obj):
    """Absolute px coords for a Fabric polygon object (or None)."""
    if obj.get("type") != "polygon":
        return None
    pts = obj.get("points") or []
    if len(pts) < 3:
        return None
    left = float(obj.get("left", 0.0)); top = float(obj.get("top", 0.0))
    sx = float(obj.get("scaleX", 1.0)); sy = float(obj.get("scaleY", 1.0))
    po = obj.get("pathOffset", {"x": 0.0, "y": 0.0})
    po_x = float(po.get("x", 0.0)); po_y = float(po.get("y", 0.0))
    out = []
    for p in pts:
        x = left + (float(p["x"]) - po_x) * sx
        y = top  + (float(p["y"]) - po_y) * sy
        out.append((x, y))
    return np.array(out, dtype=float)

def color_to_rgb(s):
    """Parse named/hex/rgb/rgba color to (r,g,b) or None."""
    if not s: return None
    s = s.strip().lower()
    if s == "blue": return (0, 0, 255)
    if s == "red":  return (255, 0, 0)
    if s.startswith("#"):
        h = s[1:]
        if len(h) == 3: r,g,b = [int(ch*2, 16) for ch in h]
        elif len(h) == 6: r = int(h[0:2],16); g = int(h[2:4],16); b = int(h[4:6],16)
        else: return None
        return (r,g,b)
    if s.startswith("rgb"):
        nums = re.findall(r"[\d.]+", s)
        if len(nums) >= 3:
            return tuple(int(float(n)) for n in nums[:3])
    return None

def near_rgb(c, target, tol=80):
    if c is None: return False
    dr = c[0]-target[0]; dg = c[1]-target[1]; db = c
