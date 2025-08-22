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
CANVAS_W = 360
CANVAS_H = 360
MM_SPAN   = 20.0                         # drawing corresponds to [-10, +10] mm
PX_PER_MM = CANVAS_W / MM_SPAN
GRID      = (256, 256)                   # raster grid for masks/metrics
RESAMPLE_N = 400                         # perimeter resampling count

# Persist last results (plots only refresh on Go!)
st.session_state.setdefault("draw_results", None)
# Persist canvas initial drawing JSON + seed so we can recolor and remount cleanly
st.session_state.setdefault("single_seed", 0)
st.session_state.setdefault("canvas_init_json", None)
st.session_state.setdefault("poly_sig", None)  # geometry signature of polygons we last applied

# --------------------------- helpers -----------------------------------------
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
    bar_px = float(10.0 * PX_PER_MM); margin = 10.0
    y0 = float(height) - margin; x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0,"y1":y0-5,"x2":x0,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0+bar_px,"y1":y0-5,"x2":x0+bar_px,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
    ]
    return {"objects": objs}

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

def extract_polygons(json_data):
    """Return list of (fabric_obj, Nx2 array) for each polygon in drawing order."""
    results = []
    if not json_data:
        return results
    for obj in json_data.get("objects", []):
        if obj.get("type") != "polygon":
            continue
        P = _polygon_points_from_fabric(obj)
        if P is not None and len(P) >= 3:
            results.append((obj, P))
    return results

def polys_signature(polys_objs):
    """Stable signature of the first few polygons (count + coarse geometry)."""
    sig = []
    for (_, P) in polys_objs[:4]:
        # length + rounded sum for stability
        sig.append((len(P), round(float(P[:,0].sum()+P[:,1].sum()), 3)))
    return tuple(sig)

def recolor_initial(polys_objs):
    """Return initial_drawing JSON with grid + recolored polygons (A blue, B red, others grey)."""
    init = grid_objects()
    colors = [
        ("rgba(0, 0, 255, 0.20)", "blue"),        # A
        ("rgba(255, 0, 0, 0.20)", "red"),         # B
    ]
    objs = init["objects"]
    for i, (obj, _) in enumerate(polys_objs):
        # copy a lean polygon object
        poly = {k: obj[k] for k in obj.keys() if k != "data"}
        if i < 2:
            fill, stroke = colors[i]
        else:
            fill, stroke = ("rgba(128, 128, 128, 0.15)", "#888888")
        poly["fill"] = fill
        poly["stroke"] = stroke
        poly["strokeWidth"] = 2
        poly["selectable"] = True
        poly["evented"] = True
        objs.append(poly)
    return init

def mask_from_polygon(P, grid_shape):
    if P is None: return np.zeros(grid_shape, dtype=bool)
    H, W = grid_shape
    xs = P[:, 0] / (CANVAS_W - 1) * (W - 1)
    ys = P[:, 1] / (CANVAS_H - 1) * (H - 1)
    rr, cc = skpolygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, 16)
    return mask

def perimeter_points(mask, n_points=RESAMPLE_N):
    if mask is None or mask.sum() == 0: return np.zeros((0, 2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs: return np.zeros((0, 2))
    longest = max(cs, key=lambda c: len(c))
    if len(longest) < 3: return np.zeros((0, 2))
    diffs = np.diff(longest, axis=0); seglen = np.sqrt((diffs**2).sum(1))
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

# --------------------------- single canvas -----------------------------------
st.subheader("Draw two closed polygons: first = A (blue), second = B (red)")
mode = st.radio("Mode", ["Draw", "Transform"], horizontal=True, index=0)

# 1) use our stored initial_drawing (grid only on very first run)
if st.session_state.canvas_init_json is None:
    st.session_state.canvas_init_json = grid_objects()

canvas = st_canvas(
    fill_color="rgba(0, 0, 255, 0.20)",     # used while drawing; we'll recolor after
    stroke_width=2,
    stroke_color="black",
    background_color="white",
    update_streamlit=True,
    height=CANVAS_H, width=CANVAS_W,
    drawing_mode=("polygon" if mode == "Draw" else "transform"),
    initial_drawing=st.session_state.canvas_init_json,
    display_toolbar=True,
    key=f"single_canvas_{st.session_state.single_seed}",
)

# 2) recolor polygons live -> update stored initial_drawing -> remount if geometry changed
if canvas.json_data is not None:
    polys_objs = extract_polygons(canvas.json_data)
    sig = polys_signature(polys_objs)
    if sig != st.session_state.poly_sig:
        st.session_state.poly_sig = sig
        st.session_state.canvas_init_json = recolor_initial(polys_objs)
        st.session_state.single_seed += 1
        st.experimental_rerun()

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
    polys_objs = extract_polygons(canvas.json_data or {})
    if len(polys_objs) < 2:
        st.session_state.draw_results = None
        st.error("Please draw **two** closed polygons in the canvas (first = blue A, second = red B).")
    else:
        P_A = polys_objs[0][1]  # first
        P_B = polys_objs[1][1]  # second

        mA = mask_from_polygon(P_A, GRID)
        mB = mask_from_polygon(P_B, GRID)

        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)
        if len(pA) == 0 or len(pB) == 0:
            st.session_state.draw_results = None
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
    st.info("Draw two polygons in the single box (first → blue A, second → red B), then press **Go!**. "
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
