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
GRID = (256, 256)        # raster grid for masks / union preview
RESAMPLE_N = 400         # perimeter resampling count

# --------------------------- State -------------------------------------------
def _ensure_side_state(side_key: str):
    if f"{side_key}_add" not in st.session_state: st.session_state[f"{side_key}_add"] = []   # list of (N,2) px
    if f"{side_key}_sub" not in st.session_state: st.session_state[f"{side_key}_sub"] = []
    if f"{side_key}_working" not in st.session_state: st.session_state[f"{side_key}_working"] = {"objects": []}
    if f"{side_key}_seed" not in st.session_state: st.session_state[f"{side_key}_seed"] = 0  # bump to remount canvas
    if "draw_results" not in st.session_state: st.session_state.draw_results = None

_ensure_side_state("A")
_ensure_side_state("B")

# ------------------------- Grid + committed outlines -------------------------
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

def outline_lines_from_polygon(P, color="#1d4ed8", dash=None, width=2, role="committed"):
    """Convert a polygon (N,2) into Fabric line segments."""
    objs = []
    if P is None or len(P) < 2: return objs
    P = np.asarray(P, dtype=float)
    for i in range(len(P)):
        x1, y1 = P[i]
        x2, y2 = P[(i + 1) % len(P)]
        objs.append({
            "type":"line","x1":float(x1),"y1":float(y1),"x2":float(x2),"y2":float(y2),
            "stroke":color,"strokeWidth":width,
            "strokeDashArray": dash if dash else None,
            "selectable":False,"evented":False,"excludeFromExport":True,
            "data":{"role":role}
        })
    return objs

def union_outline_objects(add_polys, sub_polys, color="#1d4ed8"):
    """Compute union(add) - union(sub) on GRID and return an outline as Fabric lines."""
    mask = mask_from_polylists(add_polys, sub_polys, GRID)
    if mask.sum() == 0:
        return []
    contours = measure.find_contours(mask.astype(float), 0.5)
    objs = []
    for c in contours:
        ys, xs = c[:,0], c[:,1]
        # map from GRID to canvas px
        x_px = xs / (GRID[1]-1) * (CANVAS_W-1)
        y_px = ys / (GRID[0]-1) * (CANVAS_H-1)
        poly = np.column_stack([x_px, y_px])
        objs.extend(outline_lines_from_polygon(poly, color=color, width=3, role="preview"))
    return objs

def compose_canvas_json(side_key: str, working_json):
    """Grid + committed union preview + current working polygons (editable)."""
    color_add = "#1d4ed8" if side_key == "A" else "#dc2626"   # blue / red
    objs = []
    objs += grid_objects()
    # committed union preview (solid outline)
    objs += union_outline_objects(st.session_state[f"{side_key}_add"],
                                  st.session_state[f"{side_key}_sub"],
                                  color=color_add)
    # working polygons (filled, editable)
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
    if obj.get("type") != "polygon": return None
    pts = obj.get("points") or []
    if not pts: return None
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
    """Return a clean working JSON with only user-editable polygons (ignore grid/preview)."""
    if not json_data: return {"objects": []}
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

# ------------------------------- Masks & metrics -----------------------------
def mask_from_polylists(add_polys, sub_polys, grid_shape):
    """Raster union(add_polys) minus union(sub_polys) on GRID."""
    H, W = grid_shape
    mask = np.zeros((H, W), dtype=bool)
    # union of adds
    for P in add_polys:
        xs = P[:, 0] / (CANVAS_W - 1) * (W - 1)
        ys = P[:, 1] / (CANVAS_H - 1) * (H - 1)
        rr, cc = skpolygon(ys, xs, shape=(H, W))
        mask[rr, cc] = True
    # subtract
    if sub_polys:
        sub_mask = np.zeros_like(mask)
        for P in sub_polys:
            xs = P[:, 0] / (CANVAS_W - 1) * (W - 1)
            ys = P[:, 1] / (CANVAS_H - 1) * (H - 1)
            rr, cc = skpolygon(ys, xs, shape=(H, W))
            sub_mask[rr, cc] = True
        mask = np.logical_and(mask, ~sub_mask)

    # Clean small artifacts and fill holes
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
        ["Draw + (Add)", "Draw − (Subtract)", "Transform"],
        index=0, horizontal=True, key=f"mode_{side_key}"
    )

    # Canvas key with seed => remount on commit/reset so scene reflects committed shapes
    canvas_key = f"canvas_{side_key}_{st.session_state[f'{side_key}_seed']}"

    init_json = compose_canvas_json(side_key, st.session_state[f"{side_key}_working"])

    canvas = st_canvas(
        fill_color=stroke_fill,
        stroke_width=2,
        stroke_color=("blue" if side_key == "A" else "red"),
        background_color="white",
        update_streamlit=False,
        height=CANVAS_H, width=CANVAS_W,
        drawing_mode=("polygon" if "Draw" in mode else "transform"),
        display_toolbar=True,
        initial_drawing=init_json,
        key=canvas_key,
    )

    # Only keep user-editable polygons as "working"
    if canvas.json_data is not None:
        st.session_state[f"{side_key}_working"] = extract_working_polygons(canvas.json_data)

    cols = st.columns([1,1,1,3])
    with cols[0]:
        if st.button(f"Commit Add ({side_key})", key=f"commit_add_{side_key}"):
            new_polys = polys_from_working_json(st.session_state[f"{side_key}_working"])
            st.session_state[f"{side_key}_add"].extend(new_polys)
            st.session_state[f"{side_key}_working"] = {"objects": []}
            st.session_state[f"{side_key}_seed"] += 1  # force remount to clear working and show preview
            st.rerun()
    with cols[1]:
        if st.button(f"Commit Subtract ({side_key})", key=f"commit_sub_{side_key}"):
            new_polys = polys_from_working_json(st.session_state[f"{side_key}_working"])
            st.session_state[f"{side_key}_sub"].extend(new_polys)
            st.session_state[f"{side_key}_working"] = {"objects": []}
            st.session_state[f"{side_key}_seed"] += 1
            st.rerun()
    with cols[2]:
        if st.button(f"Reset Shape ({side_key})", key=f"reset_{side_key}"):
            st.session_state[f"{side_key}_add"] = []
            st.session_state[f"{side_key}_sub"] = []
            st.session_state[f"{side_key}_working"] = {"objects": []}
            st.session_state[f"{side_key}_seed"] += 1
            st.rerun()

    # helpful counters
    n_add = len(st.session_state[f"{side_key}_add"])
    n_sub = len(st.session_state[f"{side_key}_sub"])
    n_work = len(polys_from_working_json(st.session_state[f"{side_key}_working"]))
    st.caption(
        f"Committed: **+{n_add}** / **−{n_sub}**  |  Working (not yet committed): **{n_work}**. "
        "Draw polygons, then **Commit Add** to union them, or **Commit Subtract** to carve them out. "
        "Use **Transform** to move/scale/rotate working polygons before committing. "
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
if go:
    mA = mask_from_polylists(st.session_state["A_add"], st.session_state["A_sub"], GRID)
    mB = mask_from_polylists(st.session_state["B_add"], st.session_state["B_sub"], GRID)

    if mA.sum() == 0 or mB.sum() == 0:
        st.error("Both sides must contain at least one committed polygon (after Add/Subtract).")
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
    st.info("Draw or edit contours with **Add/Subtract**, then press **Go!** to compute and render plots. "
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
