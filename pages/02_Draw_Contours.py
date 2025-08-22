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
MM_SPAN = 20.0
PX_PER_MM = CANVAS_W / MM_SPAN
GRID = (256, 256)
RESAMPLE_N = 400

# --------------------------- State -------------------------------------------
def _ensure_side_state(side_key: str):
    # The 'Reference' contour (the committed final version)
    if f"{side_key}_ref" not in st.session_state: st.session_state[f"{side_key}_ref"] = None
    # 'New' polygons drawn on the canvas that are not yet committed
    if f"{side_key}_new_polys" not in st.session_state: st.session_state[f"{side_key}_new_polys"] = []
    # A seed to force canvas re-rendering
    if f"{side_key}_seed" not in st.session_state: st.session_state[f"{side_key}_seed"] = 0

_ensure_side_state("A"); _ensure_side_state("B")
if "draw_results" not in st.session_state: st.session_state.draw_results = None

# ------------------------- Grid & preview objects ----------------------------
# NOTE: The helper functions from your original script (grid_objects, outline_lines_from_polygon,
# _polygon_points_from_fabric, polys_from_working_json, etc.) are all correct and
# do not need to be changed. I am including the key refactored functions below.

def grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1):
    # This function is correct, no changes needed.
    # ... (same as your original code) ...
    objs = []
    step_minor = PX_PER_MM * minor
    step_major = PX_PER_MM * major
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({"type":"line","x1":xi,"y1":0,"x2":xi,"y2":float(height),
                     "stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        x += step_minor
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({"type":"line","x1":0,"y1":yi,"x2":float(width),"y2":yi,
                     "stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        y += step_minor
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({"type":"line","x1":xi,"y1":0,"x2":xi,"y2":float(height),
                     "stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        x += step_major
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({"type":"line","x1":0,"y1":yi,"x2":float(width),"y2":yi,
                     "stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        y += step_major
    objs.append({"type":"rect","left":0,"top":0,"width":float(width),"height":float(height),
                 "fill":"","stroke":"#c8c8c8","strokeWidth":1,"selectable":False,
                 "evented":False,"excludeFromExport":True,"data":{"role":"grid"}})
    bar_px = float(10.0 * PX_PER_MM); margin = 10.0
    y0 = float(height) - margin; x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True,"data":{"role":"grid"}},
        {"type":"line","x1":x0,"y1":y0-5,"x2":x0,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True,"data":{"role":"grid"}},
        {"type":"line","x1":x0+bar_px,"y1":y0-5,"x2":x0+bar_px,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True,"data":{"role":"grid"}},
    ]
    return objs

def outline_lines_from_polygon(P, color="#1d4ed8", width=3):
    # This function is correct, no changes needed.
    # ... (same as your original code) ...
    objs = []
    if P is None or len(P) < 2: return objs
    P = np.asarray(P, dtype=float)
    for i in range(len(P)):
        x1,y1 = P[i]; x2,y2 = P[(i+1) % len(P)]
        objs.append({"type":"line","x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
                     "stroke":color,"strokeWidth":width,
                     "selectable":False,"evented":False,"excludeFromExport":True,
                     "data":{"role":"ref"}})
    return objs

def _polygon_points_from_fabric(obj):
    # This function is correct, no changes needed.
    # ... (same as your original code) ...
    if obj.get("type") != "polygon": return None
    pts = obj.get("points") or []
    if not pts: return None
    left = float(obj.get("left", 0.0)); top = float(obj.get("top", 0.0))
    sx = float(obj.get("scaleX", 1.0)); sy = float(obj.get("scaleY", 1.0))
    po = obj.get("pathOffset", {"x":0.0,"y":0.0})
    po_x = float(po.get("x", 0.0)); po_y = float(po.get("y", 0.0))
    out = []
    for p in pts:
        x = left + (float(p["x"]) - po_x) * sx
        y = top  + (float(p["y"]) - po_y) * sy
        out.append((x, y))
    arr = np.array(out, dtype=float)
    return arr if len(arr) >= 3 else None

def polys_from_json_data(json_data):
    # Renamed for clarity from polys_from_working_json
    polys = []
    if not json_data or "objects" not in json_data:
        return polys
    for o in json_data["objects"]:
        # We only care about polygons drawn by the user, not grid lines or reference outlines
        if o.get("type") == "polygon" and (o.get("data") is None or o.get("data", {}).get("role") is None):
            P = _polygon_points_from_fabric(o)
            if P is not None and len(P) >= 3:
                polys.append(P)
    return polys

# ------------------------------- Boolean ops & Metrics ---------------------------------
# All boolean and metric functions are correct as they were.
# (mask_from_polygon, poly_from_mask, apply_add_subtract, perimeter_points, etc.)
def mask_from_polygon(P, grid_shape):
    if P is None: return np.zeros(grid_shape, dtype=bool)
    H,W = grid_shape
    xs = P[:,0] / (CANVAS_W-1) * (W-1)
    ys = P[:,1] / (CANVAS_H-1) * (H-1)
    rr, cc = skpolygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool); mask[rr, cc] = True
    return mask

def poly_from_mask(mask):
    if mask is None or mask.sum() == 0: return None
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, 16)
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs: return None
    longest = max(cs, key=lambda c: len(c))
    ys,xs = longest[:,0], longest[:,1]
    x_px = xs / (GRID[1]-1) * (CANVAS_W-1)
    y_px = ys / (GRID[0]-1) * (CANVAS_H-1)
    return np.column_stack([x_px, y_px])

def apply_add_subtract(ref_poly, add_polys, sub_polys):
    base = mask_from_polygon(ref_poly, GRID) if ref_poly is not None else np.zeros(GRID, dtype=bool)
    for P in add_polys: base = np.logical_or(base, mask_from_polygon(P, GRID))
    for P in sub_polys: base = np.logical_and(base, ~mask_from_polygon(P, GRID))
    return poly_from_mask(base)

def perimeter_points(mask, n_points=RESAMPLE_N):
    # This function is correct, no changes needed.
    # ... (same as your original code) ...
    if mask is None or mask.sum() == 0: return np.zeros((0,2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs: return np.zeros((0,2))
    longest = max(cs, key=lambda c: len(c))
    if len(longest) < 3: return np.zeros((0,2))
    diffs = np.diff(longest, axis=0); seglen = np.sqrt((diffs**2).sum(1))
    arclen = np.concatenate([[0], np.cumsum(seglen)])
    if arclen[-1] == 0: return np.zeros((0,2))
    s = np.linspace(0, arclen[-1], n_points, endpoint=False)
    resampled = np.zeros((n_points,2), dtype=float)
    j = 0
    for i, si in enumerate(s):
        while j < len(arclen)-1 and arclen[j+1] < si: j += 1
        t = (si - arclen[j]) / max(arclen[j+1] - arclen[j], 1e-9)
        resampled[i] = longest[j]*(1-t) + longest[j+1]*t
    ys,xs = resampled[:,0], resampled[:,1]
    x_mm = (xs / (GRID[1]-1)) * 20 - 10
    y_mm = (ys / (GRID[0]-1)) * 20 - 10
    return np.column_stack([x_mm, y_mm])

def nn_distances(P, Q):
    # This function is correct, no changes needed.
    # ... (same as your original code) ...
    if len(P)==0 or len(Q)==0:
        return np.full((len(P),), np.inf), np.full((len(Q),), np.inf)
    kdP, kdQ = cKDTree(P), cKDTree(Q)
    return kdQ.query(P, k=1, workers=-1)[0], kdP.query(Q, k=1, workers=-1)[0]

def dice_jaccard_from_masks(A, B):
    # This function is correct, no changes needed.
    # ... (same as your original code) ...
    A = A.astype(bool); B = B.astype(bool)
    inter = np.logical_and(A, B).sum()
    a = A.sum(); b = B.sum()
    union = a + b - inter
    dice = (2*inter)/(a+b) if (a+b)>0 else 0.0
    jacc = inter/union if union>0 else 0.0
    return dice, jacc, int(a), int(b), int(inter)

# -------------------------- REFACTORED UI & LOGIC ----------------------------

def compose_canvas_json(side_key: str):
    """Generates the background objects for the canvas (grid + reference)."""
    color_ref = "#1d4ed8" if side_key == "A" else "#dc2626"
    objs = grid_objects()
    objs += outline_lines_from_polygon(st.session_state[f"{side_key}_ref"], color=color_ref, width=3)
    return {"objects": objs}

def canvas_section(side_key: str, stroke_fill: str):
    st.subheader(f"Contour {side_key}")
    mode = st.radio(f"Mode ({side_key})", ["Draw", "Transform"], index=0, horizontal=True, key=f"mode_{side_key}")

    canvas_key = f"canvas_{side_key}_{st.session_state[f'{side_key}_seed']}"

    # The canvas only displays the grid and the committed reference. It's a clean slate for drawing.
    canvas = st_canvas(
        fill_color=stroke_fill,
        stroke_width=2,
        stroke_color=("#1d4ed8" if side_key == "A" else "#dc2626"),
        background_color="white",
        update_streamlit=True,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode=("polygon" if mode == "Draw" else "transform"),
        display_toolbar=True,
        initial_drawing=compose_canvas_json(side_key),
        key=canvas_key,
    )

    # If the user has drawn something, we catch it and store it in our temporary state.
    if canvas.json_data and canvas.json_data.get("objects"):
        drawn_polys = polys_from_json_data(canvas.json_data)
        if drawn_polys:
            st.session_state[f"{side_key}_new_polys"] = drawn_polys

    cols = st.columns([1.2, 1.4, 1.2, 3])
    with cols[0]:
        if st.button(f"Commit Add ({side_key})", key=f"commit_add_{side_key}"):
            new_polys = st.session_state[f"{side_key}_new_polys"]
            if not new_polys:
                st.warning("Draw a polygon before committing.")
            else:
                st.session_state[f"{side_key}_ref"] = apply_add_subtract(
                    st.session_state[f"{side_key}_ref"], new_polys, []
                )
                st.session_state[f"{side_key}_new_polys"] = []
                st.session_state[f"{side_key}_seed"] += 1
                st.rerun()

    with cols[1]:
        if st.button(f"Commit Subtract ({side_key})", key=f"commit_sub_{side_key}"):
            new_polys = st.session_state[f"{side_key}_new_polys"]
            if st.session_state[f"{side_key}_ref"] is None:
                st.warning("Cannot subtract without a Reference contour.")
            elif not new_polys:
                st.warning("Draw a polygon to subtract before committing.")
            else:
                st.session_state[f"{side_key}_ref"] = apply_add_subtract(
                    st.session_state[f"{side_key}_ref"], [], new_polys
                )
                st.session_state[f"{side_key}_new_polys"] = []
                st.session_state[f"{side_key}_seed"] += 1
                st.rerun()

    with cols[2]:
        if st.button(f"Reset Reference ({side_key})", key=f"reset_{side_key}"):
            st.session_state[f"{side_key}_ref"] = None
            st.session_state[f"{side_key}_new_polys"] = []
            st.session_state[f"{side_key}_seed"] += 1
            st.rerun()

    ref_set = st.session_state[f"{side_key}_ref"] is not None
    n_new = len(st.session_state[f"{side_key}_new_polys"])
    st.caption(
        f"Reference: **{'set' if ref_set else 'not set'}** | Uncommitted polygons: **{n_new}**."
    )

def effective_mask_for_side(side_key: str):
    """Gets the definitive contour mask for comparison, handling all states."""
    ref = st.session_state.get(f"{side_key}_ref")
    new_polys = st.session_state.get(f"{side_key}_new_polys", [])

    # Case 1: A reference exists and there are uncommitted drawings. This is ambiguous.
    if ref is not None and new_polys:
        return None, "You have uncommitted changes. Please 'Commit' or 'Reset' them.", None

    # Case 2: A reference exists and the drawing canvas is clear. Use the reference.
    if ref is not None:
        return mask_from_polygon(ref, GRID), None, ref

    # Case 3: No reference, but new polygons have been drawn. Merge and use them.
    if new_polys:
        merged_poly = apply_add_subtract(None, new_polys, [])
        return mask_from_polygon(merged_poly, GRID), None, merged_poly

    # Case 4: Nothing is committed and nothing has been drawn.
    return None, "Draw a polygon to compare.", None

# ----------------------------- Render UI and Main Logic -----------------------------

left_col, right_col = st.columns(2)
with left_col:  canvas_section("A", "rgba(0, 0, 255, 0.20)")
with right_col: canvas_section("B", "rgba(255, 0, 0, 0.20)")

st.markdown("---")
thr = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)
go_col, clear_col, _ = st.columns([1,1,6])
go = go_col.button("Go! 🚀")
if clear_col.button("Clear plots"):
    st.session_state.draw_results = None

if go:
    mA, errA, adoptA = effective_mask_for_side("A")
    mB, errB, adoptB = effective_mask_for_side("B")

    errs = []
    if errA: errs.append(f"A: {errA}")
    if errB: errs.append(f"B: {errB}")
    if errs:
        st.error(" / ".join(errs))
    else:
        # Auto-commit the newly drawn polygons so they become the reference
        if adoptA is not None and st.session_state["A_ref"] is None:
            st.session_state["A_ref"] = adoptA
            st.session_state["A_new_polys"] = []
            st.session_state["A_seed"] += 1
        if adoptB is not None and st.session_state["B_ref"] is None:
            st.session_state["B_ref"] = adoptB
            st.session_state["B_new_polys"] = []
            st.session_state["B_seed"] += 1

        # --- METRIC CALCULATIONS (unchanged) ---
        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)
        dA, dB = nn_distances(pA, pB)

        msd  = (np.mean(dA) + np.mean(dB)) / 2 if len(dA) > 0 and len(dB) > 0 else 0
        hd95 = max(np.percentile(dA, perc), np.percentile(dB, perc)) if len(dA) > 0 and len(dB) > 0 else 0
        hdmax = max(np.max(dA), np.max(dB)) if len(dA) > 0 and len(dB) > 0 else 0
        sdice = ((dA <= thr).sum() + (dB <= thr).sum()) / (len(pA) + len(pB)) if (len(pA) + len(pB)) > 0 else 0
        dice, jacc, areaA, areaB, inter = dice_jaccard_from_masks(mA, mB)

        st.session_state.draw_results = dict(
            thr=thr, perc=perc, pA=pA, pB=pB, dA=dA, dB=dB,
            msd=msd, hd95=hd95, hdmax=hdmax, sdice=sdice,
            dice=dice, jacc=jacc, areaA=areaA, areaB=areaB, inter=inter,
            mA=mA, mB=mB,
        )
        if adoptA is not None or adoptB is not None:
             st.rerun() # Rerun to update the reference contours on the canvas

# ----------------------- Show (persisted) plots -------------------------------
# NOTE: The plotting code at the end of your script is correct and does not need changes.
res = st.session_state.draw_results
if res is None:
    st.info("Draw contours in both boxes, then press **Go!** to compute and render plots.")
else:
    # ... (all your matplotlib plotting code remains here, unchanged) ...
    thr = res["thr"]; perc = res["perc"]
    pA, pB, dA, dB = res["pA"], res["pB"], res["dA"], res["dB"]
    msd, hd95, hdmax, sdice = res["msd"], res["hd95"], res["hdmax"], res["sdice"]
    dice, jacc, areaA, areaB, inter = res["dice"], res["jacc"], res["areaA"], res["areaB"], res["inter"]
    mA, mB = res["mA"], res["mB"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.set_title("Surface DICE @ Threshold", fontweight="bold")
    ax.plot(np.append(pA[:,0], pA[0,0]), np.append(pA[:,1], pA[0,1]), "#1d4ed8", lw=1, label="A")
    ax.plot(np.append(pB[:,0], pB[0,0]), np.append(pB[:,1], pB[0,1]), "#dc2626", lw=1, label="B")
    
    # Visualization of distances from B's points to A's surface
    ok_b = dB <= thr
    ax.scatter(pB[ok_b,0], pB[ok_b,1], c="green", s=12, alpha=0.85, label=f"B points ≤ {thr}mm from A")
    ax.scatter(pB[~ok_b,0], pB[~ok_b,1], c="orange", s=16, alpha=0.9, label=f"B points > {thr}mm from A")
    
    ax.text(0.02, 0.98, f"Surface DICE: {sdice:.3f}", transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1]
    ax.set_title("Surface Distance Distribution", fontweight="bold")
    all_d = np.concatenate([dA, dB]); maxd = float(np.max(all_d)) if all_d.size > 0 else 1.0
    bins = np.linspace(0, max(1.0, maxd), 30)
    ax.hist(all_d, bins=bins, alpha=0.7, color="skyblue", edgecolor="black", label="All distances (A↔B)")
    ax.axvline(msd,  color="red",    linestyle="--", label=f"Mean: {msd:.2f}")
    ax.axvline(hd95, color="orange", linestyle="--", label=f"HD{perc:.0f}: {hd95:.2f}")
    ax.axvline(hdmax,color="purple", linestyle="--", label=f"Max: {hdmax:.2f}")
    ax.axvline(thr,  color="green",  linestyle="--", label=f"Thresh: {thr:.2f}")
    ax.set_xlabel("Distance (mm)"); ax.set_ylabel("Frequency"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[2]
    ax.set_title("Volumetric Overlap (Masks)", fontweight="bold")
    # Plot filled regions for overlap visualization
    ax.imshow(mA, extent=[-10, 10, -10, 10], origin="lower", cmap="Blues", alpha=0.5)
    ax.imshow(mB, extent=[-10, 10, -10, 10], origin="lower", cmap="Reds", alpha=0.5)
    # Plot outlines
    for mask, color_name, lbl in [(mA, "#1d4ed8", "A"), (mB, "#dc2626", "B")]:
        cs = measure.find_contours(mask.astype(float), 0.5)
        if cs:
            longest = max(cs, key=lambda c: len(c))
            ys,xs = longest[:,0], longest[:,1]
            x_mm = (xs / (GRID[1]-1)) * 20 - 10
            y_mm = (ys / (GRID[0]-1)) * 20 - 10
            ax.plot(x_mm, y_mm, color_name, lw=2, label=lbl)
    ax.text(0.02, 0.98, f"DICE: {dice:.3f} | Jaccard: {jacc:.3f}\n"
                        f"Area A: {areaA} px | Area B: {areaB} px\nIntersection: {inter} px",
            transform=ax.transAxes, va="top",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8), fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(-10,10); ax.set_ylim(-10,10)
    ax.set_xlabel("X (mm)"); ax.set_ylabel("Y (mm)"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
