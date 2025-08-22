# pages/02_Draw_Contours.py

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology, color, util

st.set_page_config(layout="wide", page_title="Draw Contours - RadOnc Metrics")
st.title("Draw Two Contours and Compare")

# Canvas + processing settings
CANVAS_W = 300
CANVAS_H = 300
GRID = (256, 256)    # raster grid for masks
RESAMPLE_N = 400     # perimeter resampling count

left, right = st.columns(2)
with left:
    st.subheader("Contour A")
    canvasA = st_canvas(
        fill_color="rgba(0, 0, 255, 0.2)",
        stroke_width=2,
        stroke_color="blue",
        background_color="white",
        update_streamlit=False,     # less reruns during drawing
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="polygon",     # polygon ensures closed loops
        key="canvasA",
    )

with right:
    st.subheader("Contour B")
    canvasB = st_canvas(
        fill_color="rgba(255, 0, 0, 0.2)",
        stroke_width=2,
        stroke_color="red",
        background_color="white",
        update_streamlit=False,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="polygon",
        key="canvasB",
    )

def mask_from_canvasimg(img_rgba, grid_shape):
    """Convert canvas RGBA image to a cleaned binary mask on a fixed grid."""
    if img_rgba is None:
        return None
    # Normalize to 0..1 then RGB, then threshold any drawn pixel
    rgb = util.img_as_ubyte(color.rgba2rgb(img_rgba / 255.0))
    gray = np.mean(rgb, axis=2)
    mask = gray > 0  # any non-white pixel
    # Resize to GRID via nearest
    zy = grid_shape[0] / mask.shape[0]
    zx = grid_shape[1] / mask.shape[1]
    mask_small = ndi.zoom(mask.astype(np.uint8), (zy, zx), order=0) > 0
    # Clean: remove specks, close gaps, fill holes
    mask_small = morphology.remove_small_objects(mask_small, 16)
    mask_small = morphology.binary_closing(mask_small, morphology.disk(2))
    mask_small = morphology.binary_fill_holes(mask_small)
    return mask_small

def perimeter_points(mask, n_points=RESAMPLE_N):
    """Extract the largest closed contour and resample to n_points in (x_mm, y_mm) on [-10,10]^2."""
    if mask is None or mask.sum() == 0:
        return np.zeros((0, 2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs:
        return np.zeros((0, 2))
    longest = max(cs, key=lambda c: len(c))  # (row, col)
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
    # Map (row,col) -> (x,y) in [-10,10] mm for consistency with main page
    ys, xs = resampled[:, 0], resampled[:, 1]
    x_mm = (xs / (GRID[1] - 1)) * 20 - 10
    y_mm = (ys / (GRID[0] - 1)) * 20 - 10
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

st.markdown("---")
thr = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)
go = st.button("Go! 🚀")

if go:
    mA = mask_from_canvasimg(canvasA.image_data, GRID)
    mB = mask_from_canvasimg(canvasB.image_data, GRID)

    if mA is None or mA.sum() == 0 or mB is None or mB.sum() == 0:
        st.error("Both contours must be drawn and form closed regions.")
        st.stop()

    pA = perimeter_points(mA, RESAMPLE_N)
    pB = perimeter_points(mB, RESAMPLE_N)
    if len(pA) == 0 or len(pB) == 0:
        st.error("Could not extract a closed boundary from one or both drawings.")
        st.stop()

    # Volumetric (pixel) metrics on raster masks
    dice, jacc, areaA, areaB, inter = dice_jaccard_from_masks(mA, mB)

    # Surface distances on perimeter points
    dA, dB = nn_distances(pA, pB)
    msd = (np.mean(dA) + np.mean(dB)) / 2
    hd95 = max(np.percentile(dA, perc), np.percentile(dB, perc))
    hdmax = max(np.max(dA), np.max(dB))
    sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB))

    c1, c2 = st.columns(2)
    with c1:
        st.write(f"**DICE (vol)**: {dice:.3f} | **Jaccard**: {jacc:.3f}")
        st.write(f"**Area A**: {areaA} px | **Area B**: {areaB} px | **Intersection**: {inter} px")
    with c2:
        st.write(f"**Surface DICE @ {thr:.1f} mm**: {sdice:.3f}")
        st.write(f"**MSD**: {msd:.2f} mm | **HD{int(perc)}**: {hd95:.2f} mm | **Max HD**: {hdmax:.2f} mm")

    # --- Minimal visuals: tolerance band, histogram, overlap ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1) Tolerance band view (A as reference)
    ax = axes[0]
    ax.set_title("Surface Tolerance Band (A as Reference)", fontweight="bold")
    # band around A using radius-free approach: scatter A perimeter, scatter B colored by distance
    # draw A perimeter
    ax.plot(np.append(pA[:, 0], pA[0, 0]), np.append(pA[:, 1], pA[0, 1]), "b-", lw=1, label="A")
    # classify B
    ok = dB <= thr
    ax.scatter(pB[ok, 0], pB[ok, 1], c="green", s=12, alpha=0.8, label="B (within tol.)")
    ax.scatter(pB[~ok, 0], pB[~ok, 1], c="red", s=16, alpha=0.9, label="B (outside tol.)")
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    # 2) Distance histogram
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
    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("Frequency"); ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # 3) Overlap (pixel areas)
    ax = axes[2]
    ax.set_title("Pixel Overlap (Masks)", fontweight="bold")
    # show outlines via find_contours
    for mask, color_name, lbl in [(mA, "blue", "A"), (mB, "red", "B")]:
        cs = measure.find_contours(mask.astype(float), 0.5)
        if cs:
            longest = max(cs, key=lambda c: len(c))
            ys, xs = longest[:, 0], longest[:, 1]
            x_mm = (xs / (GRID[1] - 1)) * 20 - 10
            y_mm = (ys / (GRID[0] - 1)) * 20 - 10
            ax.plot(x_mm, y_mm, color_name, lw=1, label=lbl)
    ax.set_aspect("equal"); ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
