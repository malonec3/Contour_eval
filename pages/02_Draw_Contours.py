import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon
from PIL import Image, ImageDraw, ImageFont

# --------------------------- Page setup --------------------------------------
st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")

# Canvas + processing settings
CANVAS_W = 320
CANVAS_H = 320
GRID = (256, 256)      # raster grid for masks
RESAMPLE_N = 400       # perimeter resampling count

# Persist drawings & last results across reruns
if "draw_A_json" not in st.session_state: st.session_state.draw_A_json = None
if "draw_B_json" not in st.session_state: st.session_state.draw_B_json = None
if "draw_results" not in st.session_state: st.session_state.draw_results = None

# ------------------------- Helpers: grid + scale -----------------------------
def make_grid_background(width=CANVAS_W, height=CANVAS_H,
                         mm_span=20, major=5, minor=1, margin=8) -> Image.Image:
    """
    Build a white background with a metric grid mapped to [-mm_span/2, +mm_span/2] mm.
    Adds a 10 mm scale bar bottom-left.
    """
    px_per_mm = width / mm_span
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # minor grid
    step_minor = px_per_mm * minor
    x = 0.0
    while x < width:
        xi = int(round(x))
        draw.line([(xi, 0), (xi, height)], fill=(235, 235, 235), width=1)
        x += step_minor
    y = 0.0
    while y < height:
        yi = int(round(y))
        draw.line([(0, yi), (width, yi)], fill=(235, 235, 235), width=1)
        y += step_minor

    # major grid (darker)
    step_major = px_per_mm * major
    x = 0.0
    while x < width:
        xi = int(round(x))
        draw.line([(xi, 0), (xi, height)], fill=(210, 210, 210), width=1)
        x += step_major
    y = 0.0
    while y < height:
        yi = int(round(y))
        draw.line([(0, yi), (width, yi)], fill=(210, 210, 210), width=1)
        y += step_major

    # axes box
    draw.rectangle([(0, 0), (width-1, height-1)], outline=(200, 200, 200), width=1)

    # scale bar: 10 mm
    bar_mm = 10
    bar_px = int(round(bar_mm * px_per_mm))
    x0 = margin
    y0 = height - margin - 10
    draw.line([(x0, y0), (x0 + bar_px, y0)], fill=(0, 0, 0), width=3)
    # label
    try:
        font = ImageFont.load_default()
        draw.text((x0 + bar_px + 6, y0 - 6), f"{bar_mm} mm", fill=(0, 0, 0), font=font)
    except Exception:
        pass
    return img

GRID_BG = make_grid_background()

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

def mask_from_canvas(canvas_json, canvas_img, grid_shape):
    """
    Prefer union of all polygon objects from JSON. If none, fall back to pixel threshold.
    """
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    used_json = False

    objects = (canvas_json or {}).get("objects") or []
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

    # Fallback: any non-white pixel
    if canvas_img is None:
        return None
    nonwhite = np.any(canvas_img[:, :, :3] < 250, axis=2)
    zy = H / canvas_img.shape[0]; zx = W / canvas_img.shape[1]
    mask_small = ndi.zoom(nonwhite.astype(np.uint8), (zy, zx), order=0) > 0
    mask_small = morphology.binary_closing(mask_small, morphology.disk(2))
    mask_small = ndi.binary_fill_holes(mask_small)
    mask_small = morphology.remove_small_objects(mask_small, 16)
    return mask_small

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
        background_image=GRID_BG,      # grid + scale
        update_streamlit=False,        # update only on mouseup
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="transform",      # start in edit mode; toolbar lets user switch
        display_toolbar=True,          # users can pick Polygon/Transform/Delete etc.
        initial_drawing=st.session_state.draw_A_json,  # persist objects across reruns
        key="canvasA",
    )
with right:
    st.subheader("Contour B")
    canvasB = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=2,
        stroke_color="red",
        background_image=GRID_BG,
        update_streamlit=False,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="transform",
        display_toolbar=True,
        initial_drawing=st.session_state.draw_B_json,
        key="canvasB",
    )

# Keep latest JSON so editing persists even if we don't recompute plots
if canvasA.json_data is not None:
    st.session_state.draw_A_json = canvasA.json_data
if canvasB.json_data is not None:
    st.session_state.draw_B_json = canvasB.json_data

st.caption("Tip: Use the toolbar to **Polygon** (draw), **Transform** (move/scale/rotate), and **trash** (delete). You can add multiple polygons; we’ll union them on each side.")

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
    mA = mask_from_canvas(st.session_state.draw_A_json, canvasA.image_data, GRID)
    mB = mask_from_canvas(st.session_state.draw_B_json, canvasB.image_data, GRID)

    if mA is None or mA.sum() == 0 or mB is None or mB.sum() == 0:
        st.error("Both sides must contain at least one closed polygon (or stroke).")
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
    st.info("Draw or edit contours, then press **Go!** to compute and render plots. Edits won’t clear the previous plots until you press Go! again.")
else:
    thr = res["thr"]; perc = res["perc"]
    pA, pB, dA, dB = res["pA"], res["pB"], res["dA"], res["dB"]
    msd, hd95, hdmax, sdice = res["msd"], res["hd95"], res["hdmax"], res["sdice"]
    dice, jacc, areaA, areaB, inter = res["dice"], res["jacc"], res["areaA"], res["areaB"], res["inter"]
    mA, mB = res["mA"], res["mB"]

    # three plots: Surface DICE @ threshold, distance histogram, DICE overlap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Surface DICE @ threshold (A as reference)
    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold (A as Ref.)", fontweight="bold")
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    ok = dB <= thr
    ax.scatter(pB[ok, 0], pB[ok, 1], c="green", s=12, alpha=0.85, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red", s=16, alpha=0.9,  label="B (outside tol.)")
    ax.text(0.02, 0.98, f"Surface DICE: {sdice:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # 2) Distance distribution (unchanged)
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
