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
MM_SPAN = 20.0           # map canvas to [-10, +10] mm
PX_PER_MM = CANVAS_W / MM_SPAN
GRID = (256, 256)        # raster grid for boolean ops / metrics
RESAMPLE_N = 400         # perimeter resampling count

# --------------------------- State -------------------------------------------
def _ensure_side_state(side_key: str):
    # current reference polygon in canvas pixels (N,2) or None
    if f"{side_key}_ref" not in st.session_state:
        st.session_state[f"{side_key}_ref"] = None
    # working (uncommitted) polygons currently on the canvas (Fabric JSON)
    if f"{side_key}_working" not in st.session_state:
        st.session_state[f"{side_key}_working"] = {"objects": []}
    # bump to remount canvas after commit/reset
    if f"{side_key}_seed" not in st.session_state:
        st.session_state[f"{side_key}_seed"] = 0

_ensure_side_state("A")
_ensure_side_state("B")
if "draw_results" not in st.session_state:
    st.session_state.draw_results = None

# ------------------------- Grid & preview objects ----------------------------
def grid_objects(width=CANVAS_W, height=CANVAS_H, major=5, minor=1):
    """Non-selectable Fabric objects: metric grid + 10 mm scale bar."""
    objs = []
    step_minor = PX_PER_MM * minor
    step_major = PX_PER_MM * major

    # minor grid
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({"type":"line","x1":xi,"y1":0.0,"x2":xi,"y2":float(height),
                     "stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        x += step_minor
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({"type":"line","x1":0.0,"y1":yi,"x2":float(width),"y2":yi,
                     "stroke":"#ebebeb","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        y += step_minor

    # major grid
    x = 0.0
    while x <= width + 0.5:
        xi = float(x)
        objs.append({"type":"line","x1":xi,"y1":0.0,"x2":xi,"y2":float(height),
                     "stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        x += step_major
    y = 0.0
    while y <= height + 0.5:
        yi = float(y)
        objs.append({"type":"line","x1":0.0,"y1":yi,"x2":float(width),"y2":yi,
                     "stroke":"#d2d2d2","strokeWidth":1,"selectable":False,"evented":False,
                     "excludeFromExport":True,"data":{"role":"grid"}})
        y += step_major

    # border
    objs.append({"type":"rect","left":0,"top":0,"width":float(width),"height":float(height),
                 "fill":"","stroke":"#c8c8c8","strokeWidth":1,"selectable":False,
                 "evented":False,"excludeFromExport":True,"data":{"role":"grid"}})

    # 10 mm scale bar
    bar_px = float(10.0 * PX_PER_MM)
    margin = 10.0
    y0 = float(height) - margin
    x0 = margin
    objs += [
        {"type":"line","x1":x0,"y1":y0,"x2":x0+bar_px,"y2":y0,"stroke":"#000","strokeWidth":3,
         "selectable":False,"evented":False,"excludeFromExport":True,"data":{"role":"grid"}},
        {"type":"line","x1":x0,"y1":y0-5,"x2":x0,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True,"data":{"role":"grid"}},
        {"type":"line","x1":x0+bar_px,"y1":y0-5,"x2":x0+bar_px,"y2":y0+5,"stroke":"#000","strokeWidth":2,
         "selectable":False,"evented":False,"excludeFromExport":True,"data":{"role":"grid"}},
    ]
    return objs

def outline_lines_from_polygon(P, color="#1d4ed8", width=3, role="ref"):
    """Convert a polygon (N,2) into Fabric line segments."""
    objs = []
    if P is None or len(P) < 2:
        return objs
    P = np.asarray(P, dtype=float)
    for i in range(len(P)):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % len(P)]
        objs.append({
            "type":"line","x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
            "stroke":color,"strokeWidth":width,
            "selectable":False,"evented":False,"excludeFromExport":True,
            "data":{"role":role}
        })
    return objs

def compose_canvas_json(side_key: str, working_json):
    """Grid + Reference outline + current working polygons (editable)."""
    color_ref = "#1d4ed8" if side_key == "A" else "#dc2626"   # blue / red
    objs = []
    objs += grid_objects()
    objs += outline_lines_from_polygon(st.session_state[f"{side_key}_ref"], color=color_ref, width=3, role="ref")
    # working polygons (editable)
    if working_json and "objects" in working_json:
        for o in working_json["objects"]:
            if o.get("type") == "polygon":
                o["selectable"] = True
                o["evented"] = True
                objs.append(o)
    return {"objects": objs}

# ---------------------------- JSON <-> polygons ------------------------------
def _polygon_points_from_fabric(obj):
    """Fabric.js polygon -> absolute canvas pixels."""
    if obj.get("type") != "polygon":
        return None
    pts = obj.get("points") or []
    if not pts:
        return None
    left = float(obj.get("left", 0.0));  top  = float(obj.get("top", 0.0))
    sx   = float(obj.get("scaleX", 1.0)); sy  = float(obj.get("scaleY", 1.0))
    po   = obj.get("pathOffset", {"x": 0.0, "y": 0.0})
    po_x = float(po.get("x", 0.0));       po_y = float(po.get("y", 0.0))
    out = []
    for p in pts:
        x = left + (float(p["x"]) - po_x) * sx
        y = top  + (float(p["y"]) - po_y) * sy
        out.append((x, y))
    arr = np.array(out, dtype=float)
    return arr if len(arr) >= 3 else None

def extract_working_polygons(json_data):
    """Return a clean working JSON with only user-editable polygons (ignore grid/ref)."""
    if not json_data:
        return {"objects": []}
    objs = []
    for o in json_data.get("objects", []):
        if o.get("type") == "polygon" and (o.get("data") is None or o.get("data", {}).get("role") is None):
            objs.append(o)
    return {"objects": objs}

def polys_from_working_json(working_json):
    """Extract list of (N,2) arrays in canvas pixels from working JSON."""
    polys = []
    for o in (working_json or {}).get("objects", []):
        P = _polygon_points_from_fabric(o)
        if P is not None and len(P) >= 3:
            polys.append(P)
    return polys

# ------------------------------- Boolean ops ---------------------------------
def mask_from_polygon(P, grid_shape):
    """Rasterize a single polygon P (canvas px) on GRID."""
    if P is None:
        return np.zeros(grid_shape, dtype=bool)
    H, W = grid_shape
    xs = P[:, 0] / (CANVAS_W - 1) * (W - 1)
    ys = P[:, 1] / (CANVAS_H - 1) * (H - 1)
    rr, cc = skpolygon(ys, xs, shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    return mask

def poly_from_mask(mask):
    """Largest contour of mask -> polygon (canvas px)."""
    if mask is None or mask.sum() == 0:
        return None
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = ndi.binary_fill_holes(mask)
    mask = morphology.remove_small_objects(mask, 16)
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs:
        return None
    longest = max(cs, key=lambda c: len(c))
    ys, xs = longest[:, 0], longest[:, 1]
    x_px = xs / (GRID[1] - 1) * (CANVAS_W - 1)
    y_px = ys / (GRID[0] - 1) * (CANVAS_H - 1)
    return np.column_stack([x_px, y_px])

def apply_add_subtract(ref_poly, add_polys, sub_polys):
    """ref (poly) ⊕ add_polys ⊖ sub_polys -> new ref polygon (poly)."""
    base = mask_from_polygon(ref_poly, GRID) if ref_poly is not None else np.zeros(GRID, dtype=bool)
    # add
    for P in add_polys:
        base = np.logical_or(base, mask_from_polygon(P, GRID))
    # subtract
    for P in sub_polys:
        base = np.logical_and(base, ~mask_from_polygon(P, GRID))
    return poly_from_mask(base)

# ------------------------------- Metrics -------------------------------------
def perimeter_points(mask, n_points=RESAMPLE_N):
    if mask is None or mask.sum() == 0:
        return np.zeros((0, 2))
    cs = measure.find_contours(mask.astype(float), 0.5)
    if not cs:
        return np.zeros((0, 2))
    longest = max(cs, key=lambda c: len(c))
    if len(longest) < 3:
        return np.zeros((0, 2))
    diffs = np.diff(longest, axis=0); seglen = np.sqrt((diffs**2).sum(1))
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

# ------------------------------ Canvas sections ------------------------------
def canvas_section(side_key: str, stroke_fill: str):
    st.subheader(f"Contour {side_key}")
    mode = st.radio(
        f"Mode ({side_key})",
        ["Draw", "Transform"],
        index=0, horizontal=True, key=f"mode_{side_key}"
    )

    # Canvas key with seed => remount on commit/reset so scene reflects Reference immediately
    canvas_key = f"canvas_{side_key}_{st.session_state[f'{side_key}_seed']}"

    init_json = compose_canvas_json(side_key, st.session_state[f"{side_key}_working"])

    canvas = st_canvas(
        fill_color=stroke_fill,
        stroke_width=2,
        stroke_color=("blue" if side_key == "A" else "red"),
        background_color="white",
        update_streamlit=False,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode=("polygon" if mode == "Draw" else "transform"),
        display_toolbar=True,
        initial_drawing=init_json,
        key=canvas_key,
    )

    # Only keep user-editable polygons as "working"
    if canvas.json_data is not None:
        st.session_state[f"{side_key}_working"] = extract_working_polygons(canvas.json_data)

    # Buttons row
    cols = st.columns([1.2,1.4,1.0,3])
    with cols[0]:
        if st.button(f"Commit Add ({side_key})", key=f"commit_add_{side_key}"):
            # adopt first working as reference if none yet
            working = polys_from_working_json(st.session_state[f"{side_key}_working"])
            if st.session_state[f"{side_key}_ref"] is None:
                if len(working) == 0:
                    st.warning("Draw a polygon first.")
                else:
                    ref = working[0]
                    adds = working[1:]
                    st.session_state[f"{side_key}_ref"] = apply_add_subtract(ref, adds, [])
                    st.session_state[f"{side_key}_working"] = {"objects": []}
                    st.session_state[f"{side_key}_seed"] += 1
                    st.rerun()
            else:
                if len(working) == 0:
                    st.info("Nothing to add.")
                else:
                    new_ref = apply_add_subtract(st.session_state[f"{side_key}_ref"], working, [])
                    st.session_state[f"{side_key}_ref"] = new_ref
                    st.session_state[f"{side_key}_working"] = {"objects": []}
                    st.session_state[f"{side_key}_seed"] += 1
                    st.rerun()
    with cols[1]:
        if st.button(f"Commit Subtract ({side_key})", key=f"commit_sub_{side_key}"):
            working = polys_from_working_json(st.session_state[f"{side_key}_working"])
            if st.session_state[f"{side_key}_ref"] is None:
                if len(working) == 0:
                    st.warning("Draw a polygon first.")
                else:
                    # adopt first as reference, subtract any additional ones
                    ref = working[0]
                    subs = working[1:]
                    st.session_state[f"{side_key}_ref"] = apply_add_subtract(ref, [], subs)
                    st.session_state[f"{side_key}_working"] = {"objects": []}
                    st.session_state[f"{side_key}_seed"] += 1
                    st.rerun()
            else:
                if len(working) == 0:
                    st.info("Nothing to subtract.")
                else:
                    new_ref = apply_add_subtract(st.session_state[f"{side_key}_ref"], [], working)
                    st.session_state[f"{side_key}_ref"] = new_ref
                    st.session_state[f"{side_key}_working"] = {"objects": []}
                    st.session_state[f"{side_key}_seed"] += 1
                    st.rerun()
    with cols[2]:
        if st.button(f"Reset Reference ({side_key})", key=f"reset_{side_key}"):
            st.session_state[f"{side_key}_ref"] = None
            st.session_state[f"{side_key}_working"] = {"objects": []}
            st.session_state[f"{side_key}_seed"] += 1
            st.rerun()

    # status line
    ref_set = st.session_state[f"{side_key}_ref"] is not None
    n_work = len(polys_from_working_json(st.session_state[f"{side_key}_working"]))
    st.caption(
        f"Reference: **{'set' if ref_set else 'not set'}**  |  Working polygons: **{n_work}**. "
        "Draw one polygon and press **Go** to use it directly, or sculpt the Reference with "
        "**Commit Add/Subtract**. Use **Transform** to edit working polygons before committing. "
        "Grid: 1 mm minor / 5 mm major; scale bar: 10 mm."
    )

# Render both sides
left_col, right_col = st.columns(2)
with left_col:  canvas_section("A", "rgba(0, 0, 255, 0.20)")
with right_col: canvas_section("B", "rgba(255, 0, 0, 0.20)")

# ----------------------------- Controls --------------------------------------
st.markdown("---")
thr = st.slider("Distance Threshold (mm)", 0.5, 5.0, 1.0, 0.1)
perc = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, 95.0, 0.1)
go_col, clear_col, _ = st.columns([1,1,6])
go = go_col.button("Go! 🚀")
if clear_col.button("Clear plots"):
    st.session_state.draw_results = None

# ----------------------- Compute ONLY when Go! --------------------------------
def effective_mask_for_side(side_key: str):
    """Use Reference if set; otherwise accept exactly one working polygon; else error."""
    ref = st.session_state[f"{side_key}_ref"]
    working = polys_from_working_json(st.session_state[f"{side_key}_working"])

    if ref is not None:
        return mask_from_polygon(ref, GRID), None
    if len(working) == 1:
        # auto-adopt single working polygon
        m = mask_from_polygon(working[0], GRID)
        return m, None
    if len(working) == 0:
        return None, "Draw a polygon or set a Reference."
    return None, "Multiple uncommitted polygons detected. Use Commit Add/Subtract, or Reset."

if go:
    mA, errA = effective_mask_for_side("A")
    mB, errB = effective_mask_for_side("B")

    errs = []
    if errA: errs.append(f"A: {errA}")
    if errB: errs.append(f"B: {errB}")
    if errs:
        st.error(" / ".join(errs))
    else:
        pA = perimeter_points(mA, RESAMPLE_N)
        pB = perimeter_points(mB, RESAMPLE_N)
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
