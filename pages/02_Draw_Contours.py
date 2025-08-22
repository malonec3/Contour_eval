import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon

# -----------------------------------------------------------------------------
# Page setup / persistent results
# -----------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")

# Tighten column gap so canvases sit side-by-side (almost touching) + bigger buttons
st.markdown("""
<style>
/* make columns hug each other */
[data-testid="stHorizontalBlock"] { gap: 0rem !important; }
/* optional: slightly reduce spacing under subheaders above canvases */
h3, h4 { margin-bottom: .25rem !important; }
/* bigger buttons */
div.stButton > button { padding: 0.7rem 1.2rem; font-size: 1.05rem; width: 100%; }
</style>
""", unsafe_allow_html=True)

# Persist last computed plots so they don't disappear on slider/canvas edits
st.session_state.setdefault("draw_results", None)

# Canvas & processing settings (BIGGER canvases)
CANVAS_W = 480
CANVAS_H = 480
MM_SPAN   = 20.0                    # world extent [-10, +10] mm both axes
PX_PER_MM = CANVAS_W / MM_SPAN

GRID = (256, 256)                   # raster grid for masks
RESAMPLE_N = 400                    # perimeter resampling count

# -----------------------------------------------------------------------------
# Grid + 10 mm scale as Fabric objects (non-selectable and ignored by extractor)
# -----------------------------------------------------------------------------
def grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1):
    """Fabric objects for a light grid and a 10 mm scale bar."""
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
    bar_px = 10 * PX_PER_MM; margin = 12.0; y0 = height - margin; x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0,"y1":y0-6,"x2":x0,"y2":y0+6,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0+bar_px,"y1":y0-6,"x2":x0+bar_px,"y2":y0+6,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
    ]
    return objs

GRID_OBJS = grid_objects()

# -----------------------------------------------------------------------------
# Helpers (with corrected Y orientation)
# -----------------------------------------------------------------------------
def _polygon_points_from_fabric(obj):
    """Convert a Fabric.js polygon to absolute pixel coordinates."""
    if obj.get("type") != "polygon":
        return None
    pts = obj.get("points")
    if not pts:
        return None
    left   = float(obj.get("left", 0.0))
    top    = float(obj.get("top", 0.0))
    sx     = float(obj.get("scaleX", 1.0))
    sy     = float(obj.get("scaleY", 1.0))
    po     = obj.get("pathOffset", {"x": 0.0, "y": 0.0})
    po_x   = float(po.get("x", 0.0))
    po_y   = float(po.get("y", 0.0))

    out = []
    for p in pts:
        x = left + (float(p["x"]) - po_x) * sx
        y = top  + (float(p["y"]) - po_y) * sy
        out.append((x, y))
    arr = np.array(out, dtype=float)
    return arr if len(arr) >= 3 else None


def mask_from_canvas(canvas, grid_shape):
    """Prefer polygons from JSON; fallback to pixel thresholding."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)

    jd = canvas.json_data or {}
    objects = jd.get("objects") or []

    used_polygon = False
    for obj in objects:
        poly = _polygon_points_from_fabric(obj)
        if poly is None:
            continue
        used_polygon = True
        xs = poly[:, 0] / (CANVAS_W - 1) * (W - 1)
        ys = poly[:, 1] / (CANVAS_H - 1) * (H - 1)
        rr, cc = skpolygon(ys, xs, shape=(H, W))
        mask[rr, cc] = True

    if used_polygon:
        mask = morphology.binary_closing(mask, morphology.disk(2))
        mask = ndi.binary_fill_holes(mask)
        mask = morphology.remove_small_objects(mask, 16)
        return mask

    # Fallback (pixel)
    img = canvas.image_data
    if img is None:
        return None
    nonwhite = np.any(img[:, :, :3] < 250, axis=2)
    zy = H / img.shape[0]
    zx = W / img.shape[1]
    mask_small = ndi.zoom(nonwhite.astype(np.uint8), (zy, zx), order=0) > 0
    mask_small = morphology.binary_closing(mask_small, morphology.disk(2))
    mask_small = ndi.binary_fill_holes(mask_small)
    mask_small = morphology.remove_small_objects(mask_small, 16)
    return mask_small


def perimeter_points(mask, n_points=RESAMPLE_N):
    """Resample boundary to n points in **mm** with y flipped upward."""
    if mask is None or mask.sum() == 0:
        return np.zeros((0, 2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs:
        return np.zeros((0, 2))
    longest = max(cs, key=lambda c: len(c))
    if len(longest) < 3:
        return np.zeros((0, 2))

    diffs = np.diff(longest, axis=0)
    seglen = np.sqrt((diffs**2).sum(1))
    arclen = np.concatenate([[0], np.cumsum(seglen)])
    if arclen[-1] == 0:
        return np.zeros((0, 2))

    s = np.linspace(0, arclen[-1], n_points, endpoint=False)
    resampled = np.zeros((n_points, 2), dtype=float)
    j = 0
    for i, si in enumerate(s):
        while j < len(arclen) - 1 and arclen[j + 1] < si:
            j += 1
        t = (si - arclen[j]) / max(arclen[j + 1] - arclen[j], 1e-9)
        resampled[i] = longest[j] * (1 - t) + longest[j + 1] * t

    ys, xs = resampled[:, 0], resampled[:, 1]
    x_mm = (xs / (GRID[1] - 1)) * 20 - 10
    y_mm = 10 - (ys / (GRID[0] - 1)) * 20   # flipped so up is positive
    return np.column_stack([x_mm, y_mm])


def nn_distances(P, Q):
    """Nearest-neighbor distances both ways."""
    if len(P) == 0 or len(Q) == 0:
        dP = np.full((len(P),), np.inf)
        dQ = np.full((len(Q),), np.inf)
        return dP, dQ
    kdP, kdQ = cKDTree(P), cKDTree(Q)
    dP = kdQ.query(P, k=1, workers=-1)[0]
    dQ = kdP.query(Q, k=1, workers=-1)[0]
    return dP, dQ


def dice_jaccard_from_masks(A, B):
    A = A.astype(bool); B = B.astype(bool)
    inter = np.logical_and(A, B).sum()
    a = A.sum(); b = B.sum()
    union = a + b - inter
    dice = (2 * inter) / (a + b) if (a + b) > 0 else 0.0
    jacc = inter / union if union > 0 else 0.0
    return dice, jacc, int(a), int(b), int(inter)


# -----------------------------------------------------------------------------
# Draw/Transform toggle + canvases (side by side, touching)
# -----------------------------------------------------------------------------
st.markdown("**Mode**")
mode = st.radio("", ["Draw", "Transform"], horizontal=True, index=0)
drawing_mode = "polygon" if mode == "Draw" else "transform"

# headers on one row
hA, hB = st.columns(2)
with hA: st.subheader("Contour A")
with hB: st.subheader("Contour B")

# canvases on the very next row, with 0 gap (thanks to CSS above)
colA, colB = st.columns(2)
with colA:
    canvasA = st_canvas(
        fill_color="rgba(0, 0, 255, 0.20)",
        stroke_width=2,
        stroke_color="blue",
        background_color="white",
        update_streamlit=True,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode=drawing_mode,
        initial_drawing={"objects": GRID_OBJS},
        display_toolbar=True,
        key="canvasA",
    )
with colB:
    canvasB = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=2,
        stroke_color="red",
        background_color="white",
        update_streamlit=True,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode=drawing_mode,
        initial_drawing={"objects": GRID_OBJS},
        display_toolbar=True,
        key="canvasB",
    )

# -----------------------------------------------------------------------------
# Controls
# -----------------------------------------------------------------------------
st.markdown("---")
thr  = st.slider("Distance Threshold (mm)", 0.0, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)

c1, c2, _ = st.columns([1,1,6])
go = c1.button("Go! 🚀", key="go_btn")
if c2.button("Clear plots", key="clear_btn"):
    st.session_state.draw_results = None

# -----------------------------------------------------------------------------
# Compute on Go (and persist results)
# -----------------------------------------------------------------------------
if go:
    mA = mask_from_canvas(canvasA, GRID)
    mB = mask_from_canvas(canvasB, GRID)

    if mA is None or mA.sum() == 0 or mB is None or mB.sum() == 0:
        st.session_state.draw_results = None
        st.error("Both contours must be drawn and form closed regions.")
    else:
        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)
        if len(pA) == 0 or len(pB) == 0:
            st.session_state.draw_results = None
            st.error("Could not extract a closed boundary from one or both drawings.")
        else:
            dice, jacc, areaA, areaB, inter = dice_jaccard_from_masks(mA, mB)
            dA, dB = nn_distances(pA, pB)
            msd = (np.mean(dA) + np.mean(dB)) / 2
            hd95 = max(np.percentile(dA, perc), np.percentile(dB), initial=0)
            hdmax = max(np.max(dA), np.max(dB))
            sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB))

            st.session_state.draw_results = dict(
                thr=thr, perc=perc, mA=mA, mB=mB,
                pA=pA, pB=pB, dA=dA, dB=dB,
                msd=msd, hd95=hd95, hdmax=hdmax, sdice=sdice,
                dice=dice, jacc=jacc, areaA=areaA, areaB=areaB, inter=inter
            )

# -----------------------------------------------------------------------------
# Render persisted results (until next Go)
# -----------------------------------------------------------------------------
res = st.session_state.draw_results
if res is None:
    st.info("Draw a closed polygon in each box (use **Draw**). Use **Transform** to tweak it. "
            "Press **Go!** to compute metrics; plots remain until you press Go again.")
else:
    thr  = res["thr"];   perc = res["perc"]
    mA   = res["mA"];    mB   = res["mB"]
    pA   = res["pA"];    pB   = res["pB"]
    dA   = res["dA"];    dB   = res["dB"]
    msd  = res["msd"];   hd95 = res["hd95"]; hdmax = res["hdmax"]; sdice = res["sdice"]
    dice = res["dice"];  jacc = res["jacc"]; areaA = res["areaA"]; areaB = res["areaB"]; inter = res["inter"]

    c3, c4 = st.columns(2)
    with c3:
        st.write(f"**DICE (pixel)**: {dice:.3f} | **Jaccard**: {jacc:.3f}")
        st.write(f"**Area A**: {areaA} px | **Area B**: {areaB} px | **Intersection**: {inter} px")
    with c4:
        st.write(f"**Surface DICE @ {thr:.1f} mm**: {sdice:.3f}")
        st.write(f"**MSD**: {msd:.2f} mm | **HD{int(perc)}**: {hd95:.2f} mm | **Max HD**: {hdmax:.2f} mm")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Surface DICE @ threshold (A as reference)
    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold (A as Reference)", fontweight="bold")
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    ok = dB <= thr
    ax.scatter(pB[ok, 0],  pB[ok, 1],  c="green", s=12, alpha=0.85, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red",   s=16, alpha=0.9,  label="B (outside tol.)")
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    # 2) Surface distance distribution
    ax = axes[1]
    ax.set_title("Surface Distance Distribution", fontweight="bold")
    all_d = np.concatenate([dA, dB])
    maxd = float(np.max(all_d)) if all_d.size > 0 else 1.0
    bins = np.linspace(0, max(1.0, maxd), 30)
    ax.hist(all_d, bins=bins, alpha=0.7, color="skyblue", edgecolor="black", label="A↔B")
    ax.axvline(msd,  color="red",    linestyle="--", label=f"Mean: {msd:.2f}")
    ax.axvline(hd95, color="orange", linestyle="--", label=f"HD{int(perc)}: {hd95:.2f}")
    ax.axvline(hdmax,color="purple", linestyle="--",
