import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

from streamlit_drawable_canvas import st_canvas
from scipy.spatial import cKDTree
from scipy import ndimage as ndi
from skimage import measure, morphology
from skimage.draw import polygon as skpolygon

# --------------------------- Page setup --------------------------------------
st.set_page_config(layout="wide", page_title="Draw & Compare Contours")
st.title("Draw Two Contours and Compare")
st.info("INSTRUCTIONS: Draw one or more shapes in Box A and Box B, then click 'Go!' to compare them.")

# --------------------------- Constants ---------------------------------------
CANVAS_W = 320
CANVAS_H = 320
MM_SPAN = 20.0
PX_PER_MM = CANVAS_W / MM_SPAN
GRID = (256, 256)
RESAMPLE_N = 400

# --------------------------- State (Simplified) ------------------------------
if "A_canvas_data" not in st.session_state: st.session_state.A_canvas_data = None
if "B_canvas_data" not in st.session_state: st.session_state.B_canvas_data = None
if "draw_results" not in st.session_state: st.session_state.draw_results = None

# ------------------------- Helper Functions (Unchanged) ----------------------
def grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1):
    objs = []
    step_minor = PX_PER_MM * minor; step_major = PX_PER_MM * major
    for x in np.arange(0, width + 0.5, step_minor):
        objs.append({"type":"line","x1":x,"y1":0,"x2":x,"y2":height,"stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False})
    for y in np.arange(0, height + 0.5, step_minor):
        objs.append({"type":"line","x1":0,"y1":y,"x2":width,"y2":y,"stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False})
    for x in np.arange(0, width + 0.5, step_major):
        objs.append({"type":"line","x1":x,"y1":0,"x2":x,"y2":height,"stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False})
    for y in np.arange(0, height + 0.5, step_major):
        objs.append({"type":"line","x1":0,"y1":y,"x2":width,"y2":y,"stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False})
    bar_px = 10.0 * PX_PER_MM; margin = 10.0; y0 = height - margin; x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,"selectable":False},
        {"type":"line","x1":x0,"y1":y0-5,"x2":x0,"y2":y0+5,"stroke":"#000","strokeWidth":2,"selectable":False},
        {"type":"line","x1":x0+bar_px,"y1":y0-5,"x2":x0+bar_px,"y2":y0+5,"stroke":"#000","strokeWidth":2,"selectable":False},
    ]
    return objs

def _polygon_points_from_fabric(obj):
    if obj.get("type") != "polygon": return None
    pts = obj.get("points", [])
    if not pts: return None
    left, top = obj.get("left", 0.0), obj.get("top", 0.0)
    sx, sy = obj.get("scaleX", 1.0), obj.get("scaleY", 1.0)
    out = [(left + p["x"] * sx, top + p["y"] * sy) for p in pts]
    return np.array(out, dtype=float) if len(out) >= 3 else None

def polys_from_json_data(json_data):
    polys = []
    if not json_data or "objects" not in json_data: return polys
    for o in json_data["objects"]:
        if o.get("type") == "polygon":
            P = _polygon_points_from_fabric(o)
            if P is not None: polys.append(P)
    return polys

def mask_from_polygon(P, grid_shape):
    if P is None: return np.zeros(grid_shape, dtype=bool)
    H,W = grid_shape
    xs = P[:,0] / (CANVAS_W-1) * (W-1)
    ys = P[:,1] / (CANVAS_H-1) * (H-1)
    rr, cc = skpolygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool); mask[rr, cc] = True
    return mask

def apply_add_subtract(ref_poly, add_polys, sub_polys):
    base = mask_from_polygon(ref_poly, GRID) if ref_poly is not None else np.zeros(GRID, dtype=bool)
    for P in add_polys: base = np.logical_or(base, mask_from_polygon(P, GRID))
    for P in sub_polys: base = np.logical_and(base, ~mask_from_polygon(P, GRID))
    return base # Return the mask directly

def perimeter_points(mask, n_points=RESAMPLE_N):
    if mask is None or mask.sum() == 0: return np.zeros((0,2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs: return np.zeros((0,2))
    longest = max(cs, key=len)
    if len(longest) < 3: return np.zeros((0,2))
    diffs = np.diff(longest, axis=0); seglen = np.sqrt((diffs**2).sum(1))
    arclen = np.concatenate([[0], np.cumsum(seglen)])
    if arclen[-1] == 0: return np.zeros((0,2))
    s = np.linspace(0, arclen[-1], n_points, endpoint=False)
    resampled = np.column_stack([np.interp(s, arclen, longest[:,1]), np.interp(s, arclen, longest[:,0])])
    x_mm = (resampled[:,0] / (GRID[1]-1)) * MM_SPAN - (MM_SPAN/2)
    y_mm = (resampled[:,1] / (GRID[0]-1)) * MM_SPAN - (MM_SPAN/2)
    return np.column_stack([x_mm, y_mm])

def nn_distances(P, Q):
    if len(P)==0 or len(Q)==0: return np.array([]), np.array([])
    return cKDTree(Q).query(P, k=1)[0], cKDTree(P).query(Q, k=1)[0]

def dice_jaccard_from_masks(A, B):
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    dice = (2*inter)/(A.sum()+B.sum()) if (A.sum()+B.sum())>0 else 0.0
    jacc = inter/union if union>0 else 0.0
    return dice, jacc, int(A.sum()), int(B.sum()), int(inter)

# -------------------------- UI & Main Logic (Simplified) ---------------------
def canvas_section(side_key: str, stroke_fill: str):
    st.subheader(f"Contour {side_key}")
    canvas = st_canvas(
        fill_color=stroke_fill,
        stroke_width=2,
        stroke_color=("#1d4ed8" if side_key == "A" else "#dc2626"),
        background_color="white",
        update_streamlit=True,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode="polygon",
        initial_drawing={"objects": grid_objects()},
        key=f"canvas_{side_key}"
    )
    if canvas.json_data is not None:
        st.session_state[f"{side_key}_canvas_data"] = canvas.json_data

left_col, right_col = st.columns(2)
with left_col: canvas_section("A", "rgba(0, 0, 255, 0.20)")
with right_col: canvas_section("B", "rgba(255, 0, 0, 0.20)")

st.markdown("---")
thr = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)
go_col, clear_col, _ = st.columns([1,1,6])
go = go_col.button("Go! 🚀")
if clear_col.button("Clear plots"):
    st.session_state.draw_results = None

if go:
    polys_A = polys_from_json_data(st.session_state.A_canvas_data)
    polys_B = polys_from_json_data(st.session_state.B_canvas_data)

    errs = []
    if not polys_A: errs.append("Please draw a contour in Box A.")
    if not polys_B: errs.append("Please draw a contour in Box B.")

    if errs:
        st.error(" / ".join(errs))
    else:
        mA = apply_add_subtract(None, polys_A, [])
        mB = apply_add_subtract(None, polys_B, [])

        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)
        dA, dB = nn_distances(pA, pB)

        msd  = (np.mean(dA) + np.mean(dB)) / 2 if dA.size > 0 and dB.size > 0 else 0
        hd95 = max(np.percentile(dA, perc), np.percentile(dB, perc)) if dA.size > 0 and dB.size > 0 else 0
        hdmax = max(np.max(dA), np.max(dB)) if dA.size > 0 and dB.size > 0 else 0
        sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB)) if (len(pA) + len(pB)) > 0 else 0
        dice, jacc, areaA, areaB, inter = dice_jaccard_from_masks(mA, mB)

        st.session_state.draw_results = dict(
            thr=thr, perc=perc, pA=pA, pB=pB, dA=dA, dB=dB,
            msd=msd, hd95=hd95, hdmax=hdmax, sdice=sdice,
            dice=dice, jacc=jacc, areaA=areaA, areaB=areaB, inter=inter,
            mA=mA, mB=mB,
        )

# ----------------------- Plotting (Unchanged) --------------------------------
res = st.session_state.draw_results
if res:
    # This entire plotting section is the same as your last correct version
    # ... (Plotting code from previous answer goes here) ...
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold", fontweight="bold")
    ax.plot(res["pA"][:,0], res["pA"][:,1], "#1d4ed8", lw=1.5, label="A")
    ok_b = res["dB"] <= res["thr"]
    ax.scatter(res["pB"][ok_b,0], res["pB"][ok_b,1], c="green", s=12, label="B (within tol.)")
    ax.scatter(res["pB"][~ok_b,0], res["pB"][~ok_b,1], c="red", s=16, label="B (outside tol.)")
    ax.set_aspect("equal"); ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[1]
    ax.set_title("Surface Distance Distribution", fontweight="bold")
    all_d = np.concatenate([res["dA"], res["dB"]])
    ax.hist(all_d, bins=30, color="skyblue", edgecolor="black")
    ax.axvline(res["msd"], color="red", linestyle="--", label=f"Mean: {res['msd']:.2f}")
    ax.axvline(res["hd95"], color="orange", linestyle="--", label=f"HD{res['perc']:.0f}: {res['hd95']:.2f}")
    ax.grid(True, alpha=0.3); ax.legend()

    ax = axes[2]
    ax.set_title("Volumetric Overlap (Masks)", fontweight="bold")
    ax.imshow(res["mA"].T, extent=[-10, 10, -10, 10], origin="lower", cmap="Blues", alpha=0.5)
    ax.imshow(res["mB"].T, extent=[-10, 10, -10, 10], origin="lower", cmap="Reds", alpha=0.5)
    ax.text(0.02, 0.98, f"DICE: {res['dice']:.3f}\nJaccard: {res['jacc']:.3f}", transform=ax.transAxes, va="top", bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.set_aspect("equal"); ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
