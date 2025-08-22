# pages/02_Draw_Contours.py
import re
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon

# =========================== page settings ====================================
st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")

# Keep columns snug
st.markdown(
    "<style>div[data-testid='column']{padding-left:.25rem;padding-right:.25rem}</style>",
    unsafe_allow_html=True,
)

# ============================ constants =======================================
CANVAS_W = 380
CANVAS_H = 380
MM_SPAN   = 20.0              # world is [-10, +10] mm in both axes
PX_PER_MM = CANVAS_W / MM_SPAN

GRID      = (256, 256)        # raster grid for masks/metrics
RESAMPLE_N = 400              # perimeter resampling count

# Colors we will *always* use for drawing
A_FILL,  A_STROKE = "rgba(0, 0, 255, 0.20)", "blue"
B_FILL,  B_STROKE = "rgba(255, 0, 0, 0.20)", "red"

# =========================== state ============================================
st.session_state.setdefault("polys_A", [])          # list of Fabric polygon dicts (tagged A)
st.session_state.setdefault("polys_B", [])          # list of Fabric polygon dicts (tagged B)
st.session_state.setdefault("canvas_seed", 0)       # bump to force remount if needed
st.session_state.setdefault("draw_results", None)   # persisted plots

# =========================== helpers ==========================================
def make_grid_background(w=CANVAS_W, h=CANVAS_H, major_mm=5, minor_mm=1) -> Image.Image:
    """Return a PIL image with a light grid & a 10 mm scale bar."""
    img = Image.new("RGB", (w, h), "white")
    drw = ImageDraw.Draw(img)

    step_minor = PX_PER_MM * minor_mm
    step_major = PX_PER_MM * major_mm

    # minor grid
    x = 0.0
    while x <= w + 0.5:
        xi = int(round(x))
        drw.line([(xi, 0), (xi, h)], fill="#ebebeb", width=1)
        x += step_minor
    y = 0.0
    while y <= h + 0.5:
        yi = int(round(y))
        drw.line([(0, yi), (w, yi)], fill="#ebebeb", width=1)
        y += step_minor

    # major grid
    x = 0.0
    while x <= w + 0.5:
        xi = int(round(x))
        drw.line([(xi, 0), (xi, h)], fill="#d2d2d2", width=1)
        x += step_major
    y = 0.0
    while y <= h + 0.5:
        yi = int(round(y))
        drw.line([(0, yi), (w, yi)], fill="#d2d2d2", width=1)
        y += step_major

    # border
    drw.rectangle([(0, 0), (w - 1, h - 1)], outline="#c8c8c8", width=1)

    # 10 mm scale bar (bottom-left)
    bar_px = int(round(10 * PX_PER_MM))
    margin = 10
    y0 = h - margin
    x0 = margin
    drw.line([(x0, y0), (x0 + bar_px, y0)], fill="black", width=3)
    drw.line([(x0, y0 - 5), (x0, y0 + 5)], fill="black", width=2)
    drw.line([(x0 + bar_px, y0 - 5), (x0 + bar_px, y0 + 5)], fill="black", width=2)

    return img

GRID_IMG = make_grid_background()

def colorize(obj, tag):
    """Force fill/stroke & tag."""
    if tag == "A":
        obj["fill"]        = A_FILL
        obj["stroke"]      = A_STROKE
    else:
        obj["fill"]        = B_FILL
        obj["stroke"]      = B_STROKE
    obj["strokeWidth"] = 2
    data = obj.get("data", {})
    data["contour"] = tag
    obj["data"] = data

def build_initial_json():
    """Only our tagged polygons go back in (the grid is a background image)."""
    return {"objects": [*st.session_state.polys_A, *st.session_state.polys_B]}

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

def tag_new_polys_from_canvas(json_data, active_tag):
    """
    Inspect canvas objects; any polygon without data.contour gets tagged to active_tag.
    We also refresh our stored A/B polygons to exactly what the canvas has (by tag).
    """
    if not json_data:
        return

    new_A, new_B = [], []
    for obj in json_data.get("objects", []):
        if obj.get("type") != "polygon":
            continue

        tag = obj.get("data", {}).get("contour")
        if not tag:
            # Newly drawn polygon – assign to current active tag
            tag = active_tag
            colorize(obj, tag)

        # Store a clean copy of the object under the tag
        o = {k: obj[k] for k in obj.keys()}
        colorize(o, tag)
        (new_A if tag == "A" else new_B).append(o)

    st.session_state.polys_A = new_A
    st.session_state.polys_B = new_B

def mask_from_objs(objs, grid_shape):
    """Union the polygons into a raster mask."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    for obj in objs:
        P = _fabric_polygon_points(obj)
        if P is None:
            continue
        xs = P[:, 0] / (CANVAS_W - 1) * (W - 1)
        ys = P[:, 1] / (CANVAS_H - 1) * (H - 1)
        rr, cc = skpolygon(ys, xs, shape=(H, W))
        mask[rr, cc] = True
    if mask.any():
        mask = morphology.binary_closing(mask, morphology.disk(2))
        mask = ndi.binary_fill_holes(mask)
        mask = morphology.remove_small_objects(mask, 16)
    return mask

def perimeter_points(mask, n_points=RESAMPLE_N):
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
    y_mm = (ys / (GRID[0] - 1)) * 20 - 10
    return np.column_stack([x_mm, y_mm])

def nn_distances(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return np.full((len(P),), np.inf), np.full((len(Q),), np.inf)
    kdP, kdQ = cKDTree(P), cKDTree(Q)
    return kdQ.query(P, k=1, workers=-1)[0], kdP.query(Q, k=1, workers=-1)[0]

def dice_jaccard_from_masks(A, B):
    A = A.astype(bool); B = B.astype(bool)
    inter = np.logical_and(A, B).sum()
    a = A.sum(); b = B.sum()
    union = a + b - inter
    dice = (2 * inter) / (a + b) if (a + b) > 0 else 0.0
    jacc = inter / union if union > 0 else 0.0
    return dice, jacc, int(a), int(b), int(inter)

# =========================== UI: draw/transform ===============================
st.subheader("Draw two polygons in one box (A = blue, B = red). Transform to edit. Press **Go!** to compare.")

mode = st.radio("Mode", ["Draw", "Transform"], horizontal=True, index=0)
active = st.radio("Active contour (drawing color)", ["A (blue)", "B (red)"], horizontal=True, index=0)
active_tag = "A" if active.startswith("A") else "B"

# Build the drawing we want to show (grid is a background image; polygons are ours)
initial_json = build_initial_json()

canvas = st_canvas(
    fill_color=(A_FILL if active_tag == "A" else B_FILL),
    stroke_width=2,
    stroke_color=(A_STROKE if active_tag == "A" else B_STROKE),
    background_image=GRID_IMG,            # <-- static grid; never part of json objects
    update_streamlit=False,               # <-- no rerun on each mouseup (prevents flicker)
    height=CANVAS_H, width=CANVAS_W,
    drawing_mode=("polygon" if mode == "Draw" else "transform"),
    initial_drawing=initial_json,
    display_toolbar=True,
    key=f"single_canvas_{st.session_state.canvas_seed}",
)

# Tag any new polygons & refresh our A/B lists from the canvas json
if canvas.json_data is not None:
    tag_new_polys_from_canvas(canvas.json_data, active_tag)

cols = st.columns([1,1,6])
with cols[0]:
    if st.button("Reset canvas"):
        st.session_state.polys_A = []
        st.session_state.polys_B = []
        st.session_state.draw_results = None
        st.session_state.canvas_seed += 1
        st.rerun()
with cols[1]:
    st.caption("Edits persist; plots update only on **Go!**.")

# ============================ controls ========================================
st.markdown("---")
thr  = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)
c1, c2, _ = st.columns([1, 1, 6])
go = c1.button("Go! 🚀")
clear = c2.button("Clear plots")
if clear:
    st.session_state.draw_results = None

# ============================ compute on Go ===================================
if go:
    if not st.session_state.polys_A or not st.session_state.polys_B:
        st.session_state.draw_results = None
        st.error("Please draw at least one **blue** polygon (A) and one **red** polygon (B).")
    else:
        mA = mask_from_objs(st.session_state.polys_A, GRID)
        mB = mask_from_objs(st.session_state.polys_B, GRID)
        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)

        if len(pA) == 0 or len(pB) == 0:
            st.session_state.draw_results = None
            st.error("Could not extract a closed boundary from one or both contours.")
        else:
            dA, dB = nn_distances(pA, pB)
            msd  = (np.mean(dA) + np.mean(dB)) / 2
            hd95 = max(np.percentile(dA, perc), np.percentile(dB, perc))
            hdmax = max(np.max(dA), np.max(dB))
            sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB))
            dice, jacc, areaA, areaB, inter = dice_jaccard_from_masks(mA, mB)

            st.session_state.draw_results = dict(
                thr=thr, perc=perc,
                pA=pA, pB=pB, dA=dA, dB=dB,
                msd=msd, hd95=hd95, hdmax=hdmax, sdice=sdice,
                dice=dice, jacc=jacc, areaA=areaA, areaB=areaB, inter=inter,
                mA=mA, mB=mB
            )

# ============================ render (persisted) ==============================
res = st.session_state.draw_results
if res is None:
    st.info("Pick a drawing color (A = blue, B = red). Draw closed polygons. "
            "Switch to **Transform** to edit/move/scale. Press **Go!** to compute. "
            "Plots remain until the next **Go!**.")
else:
    thr  = res["thr"];   perc = res["perc"]
    pA   = res["pA"];    pB   = res["pB"]
    dA   = res["dA"];    dB   = res["dB"]
    msd  = res["msd"];   hd95 = res["hd95"]; hdmax = res["hdmax"]; sdice = res["sdice"]
    dice = res["dice"];  jacc = res["jacc"]; areaA = res["areaA"]; areaB = res["areaB"]; inter = res["inter"]
    mA   = res["mA"];    mB   = res["mB"]

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
    ax.set_title("Surface DICE @ Threshold (A as Ref.)", fontweight="bold")
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    ok = res["dB"] <= thr
    ax.scatter(pB[ok, 0],  pB[ok, 1],  c="green", s=12, alpha=0.85, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red",   s=16, alpha=0.9,  label="B (outside tol.)")
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # 2) Surface distance distribution
    ax = axes[1]
    ax.set_title("Surface Distance Distribution", fontweight="bold")
    all_d = np.concatenate([dA, dB])
    maxd = float(np.max(all_d)) if all_d.size > 0 else 1.0
    bins = np.linspace(0, max(1.0, maxd), 30)
    ax.hist(all_d, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")
    ax.axvline(msd,  color="red",    linestyle="--", label=f"Mean: {msd:.2f}")
    ax.axvline(hd95, color="orange", linestyle="--", label=f"HD{int(perc)}: {hd95:.2f}")
    ax.axvline(hdmax,color="purple", linestyle="--", label=f"Max: {hdmax:.2f}")
    ax.axvline(thr,  color="green",  linestyle="--", label=f"Thresh: {thr:.2f}")
    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("Frequency"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3) DICE overlap score (shade ONLY the overlap)
    ax = axes[2]
    ax.set_title(f"DICE Overlap Score: {dice:.3f}", fontweight="bold")

    # outlines for A & B
    for mask, color_name, lbl in [(mA, "blue", "A"), (mB, "red", "B")]:
        cs = measure.find_contours(mask.astype(float), 0.5)
        if cs:
            longest = max(cs, key=lambda c: len(c))
            ys, xs = longest[:, 0], longest[:, 1]
            x_mm = (xs / (GRID[1] - 1)) * 20 - 10
            y_mm = (ys / (GRID[0] - 1)) * 20 - 10
            ax.plot(x_mm, y_mm, color_name, lw=1, label=lbl)

    # shaded intersection
    inter_mask = np.logical_and(mA, mB)
    cs_inter = measure.find_contours(inter_mask.astype(float), 0.5)
    first = True
    for contour in cs_inter:
        if len(contour) < 3:
            continue
        ys, xs = contour[:, 0], contour[:, 1]
        x_mm = (xs / (GRID[1] - 1)) * 20 - 10
        y_mm = (ys / (GRID[0] - 1)) * 20 - 10
        ax.fill(x_mm, y_mm, alpha=0.3, color="purple", label="Overlap" if first else None)
        first = False

    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
