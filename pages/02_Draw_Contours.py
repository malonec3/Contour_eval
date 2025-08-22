import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon

# --------------------------- Page setup --------------------------------------
st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")

# Canvas + processing settings
CANVAS_W = 320
CANVAS_H = 320
MM_SPAN = 20.0          # we map canvas to [-10, +10] mm
PX_PER_MM = CANVAS_W / MM_SPAN
GRID = (256, 256)       # raster grid for masks
RESAMPLE_N = 400        # perimeter resampling count

# Persist drawings & last results across reruns
if "draw_A_json" not in st.session_state: st.session_state.draw_A_json = {"objects": []}
if "draw_B_json" not in st.session_state: st.session_state.draw_B_json = {"objects": []}
if "draw_results" not in st.session_state: st.session_state.draw_results = None

# ------------------------- Helpers: grid as Fabric objects -------------------
def grid_objects(width=CANVAS_W, height=CANVAS_H, mm_span=MM_SPAN, major=5, minor=1):
    """Return a list of non-selectable Fabric objects drawing a metric grid + scale bar."""
    objs = []
    step_minor = PX_PER_MM * minor
    step_major = PX_PER_MM * major

    # minor verticals
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({
            "type": "line", "x1": xi, "y1": 0.0, "x2": xi, "y2": float(height),
            "stroke": "#ebebeb", "strokeWidth": 1,
            "selectable": False, "evented": False, "excludeFromExport": True
        })
        x += step_minor

    # minor horizontals
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({
            "type": "line", "x1": 0.0, "y1": yi, "x2": float(width), "y2": yi,
            "stroke": "#ebebeb", "strokeWidth": 1,
            "selectable": False, "evented": False, "excludeFromExport": True
        })
        y += step_minor

    # major verticals
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({
            "type": "line", "x1": xi, "y1": 0.0, "x2": xi, "y2": float(height),
            "stroke": "#d2d2d2", "strokeWidth": 1,
            "selectable": False, "evented": False, "excludeFromExport": True
        })
        x += step_major

    # major horizontals
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({
            "type": "line", "x1": 0.0, "y1": yi, "x2": float(width), "y2": yi,
            "stroke": "#d2d2d2", "strokeWidth": 1,
            "selectable": False, "evented": False, "excludeFromExport": True
        })
        y += step_major

    # border box
    objs.append({
        "type": "rect", "left": 0, "top": 0, "width": float(width), "height": float(height),
        "fill": "", "stroke": "#c8c8c8", "strokeWidth": 1,
        "selectable": False, "evented": False, "excludeFromExport": True
    })

    # scale bar: 10 mm at bottom-left
    bar_mm = 10.0
    bar_px = float(bar_mm * PX_PER_MM)
    margin = 10.0
    y0 = float(height) - margin
    x0 = margin
    objs.append({
        "type": "line", "x1": x0, "y1": y0, "x2": x0 + bar_px, "y2": y0,
        "stroke": "#000000", "strokeWidth": 3,
        "selectable": False, "evented": False, "excludeFromExport": True
    })
    # two small end caps
    objs.append({
        "type": "line", "x1": x0, "y1": y0 - 5, "x2": x0, "y2": y0 + 5,
        "stroke": "#000000", "strokeWidth": 2,
        "selectable": False, "evented": False, "excludeFromExport": True
    })
    objs.append({
        "type": "line", "x1": x0 + bar_px, "y1": y0 - 5, "x2": x0 + bar_px, "y2": y0 + 5,
        "stroke": "#000000", "strokeWidth": 2,
        "selectable": False, "evented": False, "excludeFromExport": True
    })
    return objs

def compose_initial_json(user_json):
    """Combine non-selectable grid + user's polygons into a canvas JSON."""
    user_objs = [o for o in (user_json or {}).get("objects", []) if o.get("type") == "polygon"]
    return {"objects": grid_objects() + user_objs}

def extract_polygons_only(json_data):
    """Keep only polygon objects from canvas JSON (drop grid/scale)."""
    if not json_data:
        return {"objects": []}
    polys = [o for o in json_data.get("objects", []) if o.get("type") == "polygon"]
    return {"objects": polys}

# -------------------------- Vector → mask helpers ----------------------------
def _polygon_points_from_fabric(obj):
    """Fabric.js polygon → absolute canvas pixels."""
    if obj.get("type") != "polygon":
        return None
    pts = obj.get("points") or []
    if not pts:
        return None
    left = float(obj.get("left", 0.0))
    top = float(obj.get("top", 0.0))
    sx = float(obj.get("scaleX", 1.0))
    sy = float(obj.get("scaleY", 1.0))
    po = obj.get("pathOffset", {"x": 0.0, "y": 0.0})
    po_x = float(po.get("x", 0.0))
    po_y = float(po.get("y", 0.0))

    out = []
    for p in pts:
        x = left + (float(p["x"]) - po_x) * sx
        y = top  + (float(p["y"]) - po_y) * sy
        out.append((x, y))
    arr = np.array(out, dtype=float)
    return arr if len(arr) >= 3 else None

def mask_from_canvas_json(canvas_json, grid_shape):
    """Union of all polygons → raster mask on GRID."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    used = False
    for obj in (canvas_json or {}).get("objects", []):
        poly = _polygon_points_from_fabric(obj)
        if poly is None:
            continue
        used = True
        xs = poly[:, 0] / (CANVAS_W - 1) * (W - 1)
        ys = poly[:, 1] / (CANVAS_H - 1) * (H - 1)
        rr, cc = skpolygon(ys, xs, shape=(H, W))
        mask[rr, cc] = True
    if not used:
        return None
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, 16)
    return mask

def perimeter_points(mask, n_points=RESAMPLE_N):
    """Largest closed contour → resample to n points → map to [-10,10] mm."""
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
    if arclen[-1] == 0: return np.zeros((0, 2))
    s = np.linspace(0, arclen[-1], n_points, endpoint=False)
    resampled = np.zeros((n_points, 2), dtype=float)
    j = 0
    for i, si in enumerate(s):
        while j < len(arclen) - 1 and arclen[j + 1] < si: j += 1
        t = (si - arclen[j]) / max(arclen[j + 1] - arclen[j], 1e-9)
        resampled[i] = longest[j] * (1 - t) + longest[j + 1] * t
    ys, xs = resampled[:, 0], resampled[:, 1]
    x_mm = (xs / (GRID[1] - 1)) * 20 - 10
    y_mm = (ys / (GRID[0] - 1)) * 20 - 10
    return np.column_stack([x_mm, y_mm])

def nn_distances(P, Q):
    if len(P) == 0 or len(Q) == 0:
        dP = np.full((len(P),), np.inf); dQ = np.full((len(Q),), np.inf)
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

# -------------------------- Canvases (editable) ------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Contour A")
    canvasA = st_canvas(
        fill_color="rgba(0, 0, 255, 0.20)",
        stroke_width=2,
        stroke_color="blue",
        background_color="white",
        update_streamlit=False,        # update only on mouseup
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="transform",      # start in edit mode; toolbar lets user switch
        display_toolbar=True,          # Polygon/Transform/Delete available
        initial_drawing=compose_initial_json(st.session_state.draw_A_json),
        key="canvasA",
    )
with right:
    st.subheader("Contour B")
    canvasB = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=2,
        stroke_color="red",
        background_color="white",
        update_streamlit=False,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="transform",
        display_toolbar=True,
        initial_drawing=compose_initial_json(st.session_state.draw_B_json),
        key="canvasB",
    )

# Persist only polygon objects (drop grid) so edits survive but grid doesn’t multiply
if canvasA.json_data is not None:
    st.session_state.draw_A_json = extract_polygons_only(canvasA.json_data)
if canvasB.json_data is not None:
    st.session_state.draw_B_json = extract_polygons_only(canvasB.json_data)

st.caption(
    "Use the toolbar to **Polygon** (draw), **Transform** (move/scale/rotate), and **trash** (delete). "
    "You can add multiple polygons; we’ll union them on each side. The grid is 1 mm minor / 5 mm major; "
    "the scale bar is 10 mm."
)

# ----------------------------- Controls --------------------------------------
st.markdown("---")
thr = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)

cols = st.columns([1,1,6])
go = cols[0].button("Go! 🚀")
clear_plots = cols[1].button("Clear plots")

if clear_plots:
    st.session_state.draw_results = None

# ----------------------- Compute ONLY when Go! --------------------------------
if go:
    mA = mask_from_canvas_json(st.session_state.draw_A_json, GRID)
    mB = mask_from_canvas_json(st.session_state.draw_B_json, GRID)

    if mA is None or mA.sum() == 0 or mB is None or mB.sum() == 0:
        st.error("Both sides must contain at least one closed polygon.")
    else:
        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)

        dA, dB = nn_distances(pA, pB)
        msd = (np.mean(dA) + np.mean(dB)) / 2
        hd95 = max(np.percentile(dA, perc), np.percentile(dB, perc))
        hdmax = max(np.max(dA), np.max(dB))
        sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB))

        dice, jacc, areaA, areaB, inter = dice_jaccard_from_masks(mA, mB)

        st.session_state.draw_results = dict(
            thr=thr, perc=perc,
            pA=pA, pB=pB, dA=dA, dB=dB,
            msd=msd, hd95=hd95, hdmax=hdmax, sdice=sdice,
            dice=dice, jacc=jacc, areaA=areaA, areaB=areaB, inter=inter,
            mA=mA, mB=mB,
        )

# ----------------------- Show (persisted) plots -------------------------------
res = st.session_state.draw_results
if res is None:
    st.info("Draw or edit contours, then press **Go!** to compute and render plots. "
            "Edits won’t clear the previous plots until you press Go! again.")
else:
    thr = res["thr"]; perc = res["perc"]
    pA, pB, dA, dB = res["pA"], res["pB"], res["dA"], res["dB"]
    msd, hd95, hdmax, sdice = res["msd"], res["hd95"], res["hdmax"], res["sdice"]
    dice, jacc, areaA, areaB, inter = res["dice"], res["jacc"], res["areaA"], res["areaB"], res["inter"]
    mA, mB = res["mA"], res["mB"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Surface DICE @ threshold (A as reference)
    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold (A as Ref.)", fontweight="bold")
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    ok = dB <= thr
    ax.scatter(pB[ok, 0], pB[ok, 1], c="green", s=12, alpha=0.85, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red", s=16, alpha=0.9, label="B (outside tol.)")
    ax.text(0.02, 0.98, f"Surface DICE: {sdice:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 2) Distance distribution
    ax = axes[1]
    ax.set_title("Surface Distance Distribution", fontweight="bold")
    all_d = np.concatenate([dA, dB])
    maxd = float(np.max(all_d)) if all_d.size > 0 else 1.0
    bins = np.linspace(0, max(1.0, maxd), 30)
    ax.hist(all_d, bins=bins, alpha=0.7, color="skyblue", edgecolor="black", label="A↔B")
    ax.axvline(msd,  color="red",    linestyle="--", label=f"Mean: {msd:.2f}")
    ax.axvline(hd95, color="orange", linestyle="--", label=f"HD{int(perc)}: {hd95:.2f}")
    ax.axvline(hdmax,color="purple", linestyle="--", label=f"Max: {hdmax:.2f}")
    ax.axvline(thr,  color="green",  linestyle="--", label=f"Thresh: {thr:.2f}")
    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("Frequency"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 3) Pixel DICE overlap
    ax = axes[2]
    ax.set_title("Pixel DICE Overlap (Masks)", fontweight="bold")
    for mask, color_name, lbl in [(mA, "blue", "A"), (mB, "red", "B")]:
        cs = measure.find_contours(mask.astype(float), 0.5)
        if cs:
            longest = max(cs, key=lambda c: len(c))
            ys, xs = longest[:, 0], longest[:, 1]
            x_mm = (xs / (GRID[1] - 1)) * 20 - 10
            y_mm = (ys / (GRID[0] - 1)) * 20 - 10
            ax.plot(x_mm, y_mm, color_name, lw=1, label=lbl)
    ax.text(0.02, 0.98, f"DICE: {dice:.3f} | Jaccard: {jacc:.3f}\n"
                        f"AreaA: {areaA} px | AreaB: {areaB} px | ∩: {inter} px",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
