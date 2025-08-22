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

# --------------------------- canvas & grid settings ---------------------------
CANVAS_W = 300
CANVAS_H = 300
MM_SPAN   = 20.0                         # drawing corresponds to [-10, +10] mm
PX_PER_MM = CANVAS_W / MM_SPAN
GRID      = (256, 256)                   # raster grid for masks/metrics
RESAMPLE_N = 400                         # perimeter resampling count

# Persist last results so plots only update on "Go!"
if "draw_results" not in st.session_state:
    st.session_state.draw_results = None

def grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1):
    """Non-selectable Fabric objects: metric grid + 10 mm scale bar."""
    objs = []
    step_minor = PX_PER_MM * minor
    step_major = PX_PER_MM * major

    # minor grid
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({"type":"line","x1":xi,"y1":0,"x2":xi,"y2":float(height),
                     "stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True})
        x += step_minor
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({"type":"line","x1":0,"y1":yi,"x2":float(width),"y2":yi,
                     "stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True})
        y += step_minor

    # major grid
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({"type":"line","x1":xi,"y1":0,"x2":xi,"y2":float(height),
                     "stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True})
        x += step_major
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({"type":"line","x1":0,"y1":yi,"x2":float(width),"y2":yi,
                     "stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True})
        y += step_major

    # border
    objs.append({"type":"rect","left":0,"top":0,"width":float(width),"height":float(height),
                 "fill":"","stroke":"#c8c8c8","strokeWidth":1,"selectable":False,
                 "evented":False,"excludeFromExport":True})

    # 10 mm scale bar
    bar_px = float(10.0 * PX_PER_MM)
    margin = 10.0
    y0 = float(height) - margin
    x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0,"y1":y0-5,"x2":x0,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0+bar_px,"y1":y0-5,"x2":x0+bar_px,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
    ]
    return {"objects": objs}

# --------------------------- UI: two canvases side-by-side --------------------
left, right = st.columns([1, 1])

with left:
    st.subheader("Contour A")
    canvasA = st_canvas(
        fill_color="rgba(0, 0, 255, 0.2)",
        stroke_width=2,
        stroke_color="blue",
        background_color="white",
        update_streamlit=True,             # live JSON; plots still gated by "Go!"
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="polygon",            # polygon ensures closed loop
        initial_drawing=grid_objects(),    # add grid + scale bar
        key="canvasA_basic",
    )

with right:
    st.subheader("Contour B")
    canvasB = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        stroke_color="red",
        background_color="white",
        update_streamlit=True,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="polygon",
        initial_drawing=grid_objects(),
        key="canvasB_basic",
    )

# -------------------------- helpers ------------------------------------------
def _polygon_points_from_fabric(obj):
    if obj.get("type") != "polygon":
        return None
    pts = obj.get("points")
    if not pts:
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
    arr = np.array(out, dtype=float)
    return arr if len(arr) >= 3 else None

def mask_from_canvas(canvas, grid_shape):
    """Prefer vector polygons; fallback to pixel thresholding if needed."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)

    jd = canvas.json_data or {}
    objects = jd.get("objects") or []
    used_json = False

    for obj in objects:
        poly = _polygon_points_from_fabric(obj)
        if poly is None:
            continue
        used_json = True
        xs = poly[:, 0] / (CANVAS_W - 1) * (W - 1)
        ys = poly[:, 1] / (CANVAS_H - 1) * (H - 1)
        rr, cc = skpolygon(ys, xs, shape=(H, W))
        mask[rr, cc] = True

    if used_json:
        mask = morphology.binary_closing(mask, morphology.disk(2))
        mask = ndi.binary_fill_holes(mask)
        mask = morphology.remove_small_objects(mask, 16)
        return mask

    # fallback (rare)
    img = canvas.image_data
    if img is None:
        return None
    nonwhite = np.any(img[:, :, :3] < 250, axis=2)
    zy = H / img.shape[0]; zx = W / img.shape[1]
    mask_small = ndi.zoom(nonwhite.astype(np.uint8), (zy, zx), order=0) > 0
    mask_small = morphology.binary_closing(mask_small, morphology.disk(2))
    mask_small = ndi.binary_fill_holes(mask_small)
    mask_small = morphology.remove_small_objects(mask_small, 16)
    return mask_small

def perimeter_points(mask, n_points=RESAMPLE_N):
    if mask is None or mask.sum() == 0: return np.zeros((0, 2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs: return np.zeros((0, 2))
    longest = max(cs, key=lambda c: len(c))
    if len(longest) < 3: return np.zeros((0, 2))

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

# ----------------------------- controls --------------------------------------
st.markdown("---")
thr  = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)
c1, c2, _ = st.columns([1, 1, 6])
go = c1.button("Go! 🚀")
clear = c2.button("Clear plots")

if clear:
    st.session_state.draw_results = None

# --------------------------------- compute on Go ------------------------------
if go:
    mA = mask_from_canvas(canvasA, GRID)
    mB = mask_from_canvas(canvasB, GRID)

    if mA is None or mA.sum() == 0 or mB is None or mB.sum() == 0:
        st.error("Both contours must be drawn and form closed regions.")
    else:
        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)

        if len(pA) == 0 or len(pB) == 0:
            st.error("Could not extract a closed boundary from one or both drawings.")
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

# ------------------------------ render results (persist) ----------------------
res = st.session_state.draw_results
if res is None:
    st.info("Draw contours on the left/right, then press **Go!** to compute and render plots. "
            "Edits won’t clear the previous plots until you press Go! again.")
else:
    thr  = res["thr"];   perc = res["perc"]
    pA   = res["pA"];    pB   = res["pB"]
    dA   = res["dA"];    dB   = res["dB"]
    msd  = res["msd"];   hd95 = res["hd95"]; hdmax = res["hdmax"]; sdice = res["sdice"]
    dice = res["dice"];  jacc = res["jacc"]; areaA = res["areaA"]; areaB = res["areaB"]; inter = res["inter"]
    mA   = res["mA"];    mB   = res["mB"]

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**DICE (pixel)**: {dice:.3f} | **Jaccard**: {jacc:.3f}")
        st.write(f"**Area A**: {areaA} px | **Area B**: {areaB} px | **Intersection**: {inter} px")
    with c2:
        st.write(f"**Surface DICE @ {thr:.1f} mm**: {sdice:.3f}")
        st.write(f"**MSD**: {msd:.2f} mm | **HD{int(perc)}**: {hd95:.2f} mm | **Max HD**: {hdmax:.2f} mm")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Surface DICE @ threshold (A as reference)
    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold (A as Ref.)", fontweight="bold")
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    ok = res["dB"] <= thr
    ax.scatter(pB[ok, 0], pB[ok, 1], c="green", s=12, alpha=0.85, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red", s=16, alpha=0.9, label="B (outside tol.)")
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    # 2) Distance histogram (same as before)
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

    # 3) DICE overlap (pixel masks)
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
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
