import os
import io, base64
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon

# --------------------------- Page / styling ----------------------------------
st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")
st.markdown(
    """
<style>
[data-testid="stHorizontalBlock"] { gap: 0rem !important; }
h3, h4 { margin-bottom: .25rem !important; }
div.stButton > button { padding: 0.7rem 1.2rem; font-size: 1.05rem; width: 100%; }
</style>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("draw_results", None)

# --------------------------- Base geometry -----------------------------------
BASE_W = 480
BASE_H = 480
MM_SPAN = 20.0  # world extent [-10, +10] mm both axes
GRID = (256, 256)
RESAMPLE_N = 400

# --------------------------- Assets ------------------------------------------
ASSETS_DIR = "assets"
PELVIS_PATH = os.path.join(ASSETS_DIR, "ct_pelvis.png")
THORAX_PATH = os.path.join(ASSETS_DIR, "ct_thorax.png")

def load_ct_bg(path: str):
    """Load CT; return PIL.Image or None if not found."""
    if not path or not os.path.exists(path):
        return None
    # exif-safe in case image has orientation flags
    try:
        from PIL import ImageOps
        return ImageOps.exif_transpose(Image.open(path).convert("RGB"))
    except Exception:
        return Image.open(path).convert("RGB")

def letterbox_to_height(img: Image.Image, target_h: int) -> Image.Image:
    """Scale to target height, preserve aspect (no distortion)."""
    if img.height == 0:
        return None
    scale = target_h / float(img.height)
    new_w = int(round(img.width * scale))
    return img.resize((new_w, target_h), Image.BICUBIC)

def pil_to_data_url(img: Image.Image) -> str:
    """PNG data URL for Fabric image object."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def fabric_image_object(img: Image.Image):
    """Non-selectable background image for initial_drawing."""
    return {
        "type": "image",
        "left": 0,
        "top": 0,
        "width": float(img.width),
        "height": float(img.height),
        "src": pil_to_data_url(img),
        "scaleX": 1.0,
        "scaleY": 1.0,
        "selectable": False,
        "evented": False,
        "excludeFromExport": True,
        "opacity": 1.0,
        "crossOrigin": "anonymous",
    }

# --------------------------- Grid overlay objects ----------------------------
def grid_objects(width: int, height: int, mm_span: float = MM_SPAN, major: int = 5, minor: int = 1):
    """Fabric.js objects for a grid + 10 mm scale bar."""
    px_per_mm = width / mm_span
    step_minor = px_per_mm * minor
    step_major = px_per_mm * major

    objs = []

    def add_line(x1, y1, x2, y2, color):
        objs.append(
            {
                "type": "line",
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
                "stroke": color,
                "strokeWidth": 1,
                "selectable": False,
                "evented": False,
                "excludeFromExport": True,
            }
        )

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
    objs.append(
        {
            "type": "rect",
            "left": 0, "top": 0,
            "width": float(width), "height": float(height),
            "fill": "", "stroke": "#c8c8c8", "strokeWidth": 1,
            "selectable": False, "evented": False, "excludeFromExport": True,
        }
    )

    # 10 mm scale bar (bottom-left)
    bar_px = 10 * px_per_mm; margin = 12.0; y0 = height - margin; x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0,"y1":y0-6,"x2":x0,"y2":y0+6,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
        {"type":"line","x1":x0+bar_px,"y1":y0-6,"x2":x0+bar_px,"y2":y0+6,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True},
    ]
    return objs

# --------------------------- Helpers (math) -----------------------------------
def _polygon_points_from_fabric(obj):
    """Convert a Fabric polygon to absolute pixels."""
    if obj.get("type") != "polygon":
        return None
    pts = obj.get("points")
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
        y = top + (float(p["y"]) - po_y) * sy
        out.append((x, y))
    arr = np.array(out, dtype=float)
    return arr if len(arr) >= 3 else None

def mask_from_canvas(canvas, grid_shape):
    """Prefer vector polygons; fallback to pixel threshold."""
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
    y_mm = 10 - (ys / (GRID[0] - 1)) * 20
    return np.column_stack([x_mm, y_mm])

def nn_distances(P, Q):
    if len(P) == 0 or len(Q) == 0:
        return np.full((len(P),), np.inf), np.full((len(Q),), np.inf)
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

def centroid_mm_from_mask(M):
    idx = np.argwhere(M)
    if idx.size == 0:
        return np.array([np.nan, np.nan])
    r_mean = idx[:, 0].mean()
    c_mean = idx[:, 1].mean()
    x_mm = (c_mean / (GRID[1] - 1)) * 20 - 10
    y_mm = 10 - (r_mean / (GRID[0] - 1)) * 20
    return np.array([x_mm, y_mm])

def apl_length_mm(points_test_mm, d_test_to_ref, thr_mm):
    if len(points_test_mm) == 0 or len(d_test_to_ref) == 0:
        return 0.0
    mask = d_test_to_ref > thr_mm
    if mask.sum() == 0:
        return 0.0
    total = 0.0
    n = len(points_test_mm)
    for i in range(n):
        j = (i + 1) % n
        if mask[i] and mask[j]:
            total += float(np.linalg.norm(points_test_mm[j] - points_test_mm[i]))
    return total

# --------------------------- Instructions ------------------------------------
st.markdown("**PC only - mobile device compatibility is currently under development**")
st.markdown(
    """
### How to use
1. **Draw A (blue)** – Click to add points; right-click (or double-click) to close the loop.  
2. **Draw B (red)** – Same on the right canvas. Each canvas needs one closed loop.  
3. **Transform (optional)** – Move/scale/rotate; switch back to Draw to add points.  
4. **Click Go!** to compute metrics (plots persist until you press Go! again).  
5. **Undo/Redo/Delete** controls are under each canvas.
"""
)

# ---- ONE draw/transform radio (add a key) ----
mode = st.radio("", ["Draw", "Transform"], horizontal=True, index=0, key="mode_radio")
drawing_mode = "polygon" if mode == "Draw" else "transform"

# ---- Background choice (with a key) ----
bg_choice = st.radio(
    "Canvas background",
    ["None", "Grid", "CT: Pelvis", "CT: Thorax"],
    horizontal=True,
    index=1,
    key="bg_choice_radio",
)

# --------------------------- Decide sizes/backgrounds -------------------------
TARGET_H = 480  # fit CT vertically to this height
bg_img_pil = None
initial_objects_A = []
initial_objects_B = []

if bg_choice == "CT: Pelvis":
    bg_img_pil = load_ct_bg(PELVIS_PATH)
elif bg_choice == "CT: Thorax":
    bg_img_pil = load_ct_bg(THORAX_PATH)

if bg_img_pil is not None:
    ct_fitted = letterbox_to_height(bg_img_pil, TARGET_H)
    CANVAS_W, CANVAS_H = ct_fitted.width, ct_fitted.height
    # NOTE: no grid overlay on CT
    initial_objects_A = [fabric_image_object(ct_fitted)]
    initial_objects_B = [fabric_image_object(ct_fitted)]
elif bg_choice == "Grid":
    CANVAS_W = CANVAS_H = BASE_W  # square canvas
    gridA = grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1)
    gridB = grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1)
    initial_objects_A = gridA
    initial_objects_B = gridB
else:  # "None"
    CANVAS_W = BASE_W
    CANVAS_H = BASE_H
    initial_objects_A = []
    initial_objects_B = []

# -------------------- Headers + canvases (side-by-side) --------------------
hA, hB = st.columns(2)
with hA: st.subheader("Contour A")
with hB: st.subheader("Contour B")

colA, colB = st.columns(2)

common_canvas_kwargs = dict(
    background_color="white",
    update_streamlit=True,
    height=CANVAS_H,
    width=CANVAS_W,
    drawing_mode=drawing_mode,
    display_toolbar=True,
)

with colA:
    canvasA = st_canvas(
        fill_color="rgba(0, 0, 255, 0.20)",
        stroke_width=2,
        stroke_color="blue",
        initial_drawing={"objects": initial_objects_A} if initial_objects_A else None,
        key="canvasA",
        **common_canvas_kwargs,
    )

with colB:
    canvasB = st_canvas(
        fill_color="rgba(255, 0, 0, 0.20)",
        stroke_width=2,
        stroke_color="red",
        initial_drawing={"objects": initial_objects_B} if initial_objects_B else None,
        key="canvasB",
        **common_canvas_kwargs,
    )

# --------------------------- Controls ----------------------------------------
st.markdown("---")
thr = st.slider("Distance Threshold (mm)", 0.0, 5.0, 1.0, 0.1, key="thr_slider")
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1, key="perc_slider")

c1, c2, _ = st.columns([1, 1, 6])
go = c1.button("Go! 🚀", key="go_btn")
if c2.button("Clear plots", key="clear_btn"):
    st.session_state.draw_results = None

# --------------------------- Compute on Go -----------------------------------
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
            dice, jacc, areaA_px, areaB_px, inter_px = dice_jaccard_from_masks(mA, mB)
            dA, dB = nn_distances(pA, pB)

            msd = (np.mean(dA) + np.mean(dB)) / 2.0
            hd95 = max(float(np.percentile(dA, perc)), float(np.percentile(dB, perc)))
            hdmax = max(float(np.max(dA)), float(np.max(dB)))
            sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB))

            dx = 20.0 / (GRID[1] - 1)
            dy = 20.0 / (GRID[0] - 1)
            pix_area_mm2 = dx * dy
            areaA_mm2 = float(areaA_px * pix_area_mm2)
            areaB_mm2 = float(areaB_px * pix_area_mm2)
            inter_mm2 = float(inter_px * pix_area_mm2)

            vol_ratio = (
                min(areaA_mm2, areaB_mm2) / max(areaA_mm2, areaB_mm2)
                if max(areaA_mm2, areaB_mm2) > 0
                else 0.0
            )

            cA = centroid_mm_from_mask(mA)
            cB = centroid_mm_from_mask(mB)
            center_dist = (
                float(np.linalg.norm(cA - cB)) if np.all(np.isfinite([*cA, *cB])) else float("nan")
            )

            apl = apl_length_mm(pB, dB, thr)

            st.session_state.draw_results = dict(
                thr=thr, perc=perc, mA=mA, mB=mB, pA=pA, pB=pB, dA=dA, dB=dB,
                msd=msd, hd95=hd95, hdmax=hdmax, sdice=sdice,
                dice=dice, jacc=jacc,
                areaA_px=areaA_px, areaB_px=areaB_px, inter_px=inter_px,
                areaA_mm2=areaA_mm2, areaB_mm2=areaB_mm2, inter_mm2=inter_mm2,
                vol_ratio=vol_ratio, center_dist=center_dist, apl=apl,
            )

# --------------------------- Render persisted results ------------------------
res = st.session_state.draw_results
if res is None:
    st.info(
        "Draw a closed polygon in each box (use **Draw**). Use **Transform** to tweak it. "
        "Press **Go!** to compute metrics; plots remain until you press Go again."
    )
else:
    thr = res["thr"]; perc = res["perc"]
    mA = res["mA"];   mB = res["mB"]
    pA = res["pA"];   pB = res["pB"]
    dA = res["dA"];   dB = res["dB"]
    msd = res["msd"]; hd95 = res["hd95"]; hdmax = res["hdmax"]; sdice = res["sdice"]
    dice = res["dice"]; jacc = res["jacc"]
    areaA_px = res["areaA_px"]; areaB_px = res["areaB_px"]; inter_px = res["inter_px"]
    areaA_mm2 = res["areaA_mm2"]; areaB_mm2 = res["areaB_mm2"]; inter_mm2 = res["inter_mm2"]
    vol_ratio = res["vol_ratio"]; center_dist = res["center_dist"]; apl = res["apl"]

    g1, g2, g3 = st.columns(3)
    with g1:
        st.markdown("### Volumetric Overlap Metrics (Raster)")
        st.markdown(
            f"- **DICE Coefficient:** {dice:.4f}\n"
            f"- **Jaccard Index:** {jacc:.4f}\n"
            f"- **Volume Ratio:** {vol_ratio:.4f}"
        )
    with g2:
        st.markdown("### Surface-based Metrics (Sampled Points)")
        st.markdown(
            f"- **Surface DICE @ {thr:.1f} mm:** {sdice:.4f}\n"
            f"- **Mean Surface Distance:** {msd:.3f} mm\n"
            f"- **95th Percentile HD:** {hd95:.3f} mm\n"
            f"- **Maximum Hausdorff:** {hdmax:.3f} mm"
        )
    with g3:
        st.markdown("### Geometric Properties")
        st.markdown(
            f"- **Reference Area (A):** {areaA_mm2:.2f} mm²\n"
            f"- **Test Area (B):** {areaB_mm2:.2f} mm²\n"
            f"- **Intersection Area:** {inter_mm2:.2f} mm²\n"
            f"- **Center-to-Center Distance:** {center_dist:.3f} mm\n"
            f"- **Added Path Length (APL) @ {thr:.1f} mm:** {apl:.2f} mm"
        )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold (A as Reference)", fontweight="bold")
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    ok = dB <= thr
    ax.scatter(pB[ok, 0], pB[ok, 1], c="green", s=12, alpha=0.85, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red", s=16, alpha=0.9, label="B (outside tol.)")
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.set_title("Surface Distance Distribution", fontweight="bold")
    all_d = np.concatenate([dA, dB])
    maxd = float(np.max(all_d)) if all_d.size > 0 else 1.0
    bins = np.linspace(0, max(1.0, maxd), 30)
    ax.hist(all_d, bins=bins, alpha=0.7, color="skyblue", edgecolor="black", label="A↔B")
    ax.axvline(msd, color="red", linestyle="--", label=f"Mean: {msd:.2f}")
    ax.axvline(hd95, color="orange", linestyle="--", label=f"HD{int(perc)}: {hd95:.2f}")
    ax.axvline(hdmax, color="purple", linestyle="--", label=f"Max: {hdmax:.2f}")
    ax.axvline(thr, color="green", linestyle="--", label=f"Thresh: {thr:.2f}")
    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("Frequency")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.set_title(f"DICE Overlap Score: {dice:.3f}", fontweight="bold")
    for mask, color_name, lbl in [(mA, "blue", "A"), (mB, "red", "B")]:
        cs = measure.find_contours(mask.astype(float), 0.5)
        if cs:
            longest = max(cs, key=lambda c: len(c))
            ys, xs = longest[:, 0], longest[:, 1]
            x_mm = (xs / (GRID[1] - 1)) * 20 - 10
            y_mm = 10 - (ys / (GRID[0] - 1)) * 20
            ax.plot(x_mm, y_mm, color_name, lw=1, label=lbl)

    inter_mask = np.logical_and(mA, mB)
    cs_inter = measure.find_contours(inter_mask.astype(float), 0.5)
    first = True
    for contour in cs_inter:
        if len(contour) < 3:
            continue
        ys, xs = contour[:, 0], contour[:, 1]
        x_mm = (xs / (GRID[1] - 1)) * 20 - 10
        y_mm = 10 - (ys / (GRID[0] - 1)) * 20
        ax.fill(x_mm, y_mm, alpha=0.3, color="purple", label="Overlap" if first else None)
        first = False

    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
