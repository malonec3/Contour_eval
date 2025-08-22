# streamlit_contour_app.py

# -----------------------------------------------------------------------------
# RadOnc Contour Overlap Metrics - Educational Tool
#
# Developer: Ciaran Malone
# LinkedIn: https://www.linkedin.com/in/ciaranmalone/
#
# License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International
#          (CC BY-NC-SA 4.0)
# License URL: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Creative Commons license for more details.
# -----------------------------------------------------------------------------

import io
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import CubicSpline
from scipy.spatial import cKDTree


# ============== FAST NEAREST-NEIGHBOR DISTANCES (cached) =====================

@st.cache_data(show_spinner=False)
def nn_distances_cached(c1_pts: np.ndarray, c2_pts: np.ndarray):
    """Return nearest-neighbor distances and indices using KDTree.
    Outputs: (c1_d, c1_to_c2_idx, c2_d, c2_to_c1_idx)
    """
    if c1_pts.size == 0 or c2_pts.size == 0:
        c1_d = np.full((len(c1_pts),), np.inf) if len(c1_pts) else np.array([])
        c2_d = np.full((len(c2_pts),), np.inf) if len(c2_pts) else np.array([])
        c1_idx = np.full((len(c1_pts),), -1, dtype=int)
        c2_idx = np.full((len(c2_pts),), -1, dtype=int)
        return c1_d, c1_idx, c2_d, c2_idx

    kd1 = cKDTree(c1_pts)
    kd2 = cKDTree(c2_pts)
    c1_d, c1_idx = kd2.query(c1_pts, k=1, workers=-1)
    c2_d, c2_idx = kd1.query(c2_pts, k=1, workers=-1)
    return c1_d, c1_idx, c2_d, c2_idx


# ====================== CORE ANALYSIS/ PLOTTING CLASS =========================

class ContourAnalysisApp:
    def __init__(self):
        self.metrics = {}  # Stores computed metrics

    def circle_intersection_area(self, r1, r2, d):
        """Intersection area of two circles (robust)."""
        if d >= r1 + r2:
            return 0.0
        if d <= abs(r1 - r2):
            return float(np.pi * min(r1, r2) ** 2)
        # Normal case
        r1_sq, r2_sq, d_sq = r1**2, r2**2, d**2
        cos1 = np.clip((d_sq + r1_sq - r2_sq) / (2 * d * r1), -1, 1)
        cos2 = np.clip((d_sq + r2_sq - r1_sq) / (2 * d * r2), -1, 1)
        ang1 = 2 * np.arccos(cos1)
        ang2 = 2 * np.arccos(cos2)
        a1 = 0.5 * r1_sq * (ang1 - np.sin(ang1))
        a2 = 0.5 * r2_sq * (ang2 - np.sin(ang2))
        return float(a1 + a2)

    def generate_circle_points(self, center, radius, num_points, noise_level, key, noise_cache):
        """Generate closed contour points for a circle with smooth 'human-like' noise."""
        cache = noise_cache[key]
        regenerate = (
            cache.get("noise_level") != noise_level
            or cache.get("num_points") != num_points
            or cache.get("base_radius") != radius
        )

        # Angles are always the same linspace for a given num_points
        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)

        if regenerate:
            if noise_level <= 0.0:
                delta_frac = np.zeros(num_points)
                delta_theta = np.zeros(num_points)
            else:
                n_knots = max(8, int(20 * noise_level))
                knot_ang = np.linspace(0, 2 * np.pi, n_knots, endpoint=False)

                # deterministic RNG from session state
                rng = st.session_state.rng
                rad_knot = rng.normal(0, 0.25 * noise_level, n_knots)  # radial noise
                th_knot = rng.normal(0, 0.10 * noise_level, n_knots)   # angular noise

                # periodic closure
                knot_ang_periodic = np.append(knot_ang, knot_ang[0] + 2 * np.pi)
                rad_knot_periodic = np.append(rad_knot, rad_knot[0])
                th_knot_periodic = np.append(th_knot, th_knot[0])

                cs_r = CubicSpline(knot_ang_periodic, rad_knot_periodic, bc_type="periodic")
                cs_th = CubicSpline(knot_ang_periodic, th_knot_periodic, bc_type="periodic")

                delta_frac = cs_r(angles)
                delta_theta = cs_th(angles)

            cache.update(
                {
                    "noise_level": noise_level,
                    "num_points": num_points,
                    "base_radius": radius,
                    "delta_frac": delta_frac,
                    "delta_theta": delta_theta,
                }
            )

        # retrieve
        delta_frac = cache["delta_frac"]
        delta_theta = cache["delta_theta"]

        r_perturbed = np.maximum(radius * (1.0 + delta_frac), 0.0)
        th_perturbed = angles + delta_theta
        x = center[0] + r_perturbed * np.cos(th_perturbed)
        y = center[1] + r_perturbed * np.sin(th_perturbed)
        return np.column_stack([x, y])

    def compute_metrics(self, params, noise_cache):
        c1_center = (params["circle1_x"], params["circle1_y"])
        c2_center = (params["circle2_x"], params["circle2_y"])
        r1_param = params["radius1"]
        r2_param = params["radius2"]
        thr = params["distance_threshold"]
        perc = params["percentile"]
        n_pts = params["num_points"]

        # Generate noisy contours
        c1_pts = self.generate_circle_points(
            c1_center, r1_param, n_pts, params["noise1"], key="c1", noise_cache=noise_cache
        )
        c2_pts = self.generate_circle_points(
            c2_center, r2_param, n_pts, params["noise2"], key="c2", noise_cache=noise_cache
        )

        # Geometric circle areas/overlap (for the pedagogical volumetric metrics)
        a1_geom = np.pi * r1_param**2
        a2_geom = np.pi * r2_param**2
        centre_d = float(np.linalg.norm(np.array(c1_center) - np.array(c2_center)))
        inter_a_geom = self.circle_intersection_area(r1_param, r2_param, centre_d)
        union_a_geom = a1_geom + a2_geom - inter_a_geom
        dice_geom = 2 * inter_a_geom / (a1_geom + a2_geom) if (a1_geom + a2_geom) > 0 else 0.0
        jaccard_geom = inter_a_geom / union_a_geom if union_a_geom > 0 else 0.0
        vol_ratio_geom = min(a1_geom, a2_geom) / max(a1_geom, a2_geom) if max(a1_geom, a2_geom) > 0 else 0.0

        # --------- FAST SURFACE DISTANCES: use nearest neighbor only ----------
        c1_d = np.array([np.inf])
        c2_d = np.array([np.inf])
        c1_to_c2_idx = np.array([], dtype=int)
        c2_to_c1_idx = np.array([], dtype=int)
        if n_pts > 0:
            c1_d, c1_to_c2_idx, c2_d, c2_to_c1_idx = nn_distances_cached(c1_pts, c2_pts)

        # Surface DICE (threshold band)
        surf_dice = 0.0
        if len(c1_pts) + len(c2_pts) > 0:
            n_ok_c1 = int((c1_d <= thr).sum()) if c1_d.size > 0 and np.isfinite(c1_d).any() else 0
            n_ok_c2 = int((c2_d <= thr).sum()) if c2_d.size > 0 and np.isfinite(c2_d).any() else 0
            surf_dice = (n_ok_c1 + n_ok_c2) / (len(c1_pts) + len(c2_pts))

        # Mean surface distance
        finite_c1 = c1_d[np.isfinite(c1_d)] if c1_d.size else np.array([])
        finite_c2 = c2_d[np.isfinite(c2_d)] if c2_d.size else np.array([])
        if finite_c1.size and finite_c2.size:
            msd = (finite_c1.mean() + finite_c2.mean()) / 2.0
        elif finite_c1.size:
            msd = float(finite_c1.mean())
        elif finite_c2.size:
            msd = float(finite_c2.mean())
        else:
            msd = 0.0

        # HD95
        if finite_c1.size and finite_c2.size:
            hd95 = float(max(np.percentile(finite_c1, perc), np.percentile(finite_c2, perc)))
        elif finite_c1.size:
            hd95 = float(np.percentile(finite_c1, perc))
        elif finite_c2.size:
            hd95 = float(np.percentile(finite_c2, perc))
        else:
            hd95 = 0.0

        # Max HD + endpoints (use indices from KDTree)
        hd_max = 0.0
        p1 = np.array([0.0, 0.0])
        p2 = np.array([0.0, 0.0])

        candidates = []
        if finite_c1.size:
            i = int(np.argmax(c1_d))
            candidates.append(("c1", float(c1_d[i]), i, int(c1_to_c2_idx[i])))
        if finite_c2.size:
            j = int(np.argmax(c2_d))
            candidates.append(("c2", float(c2_d[j]), j, int(c2_to_c1_idx[j])))

        if candidates:
            src, hd_max, i, j = max(candidates, key=lambda t: t[1])
            if src == "c1":
                if 0 <= i < len(c1_pts) and 0 <= j < len(c2_pts):
                    p1, p2 = c1_pts[i], c2_pts[j]
            else:
                if 0 <= j < len(c1_pts) and 0 <= i < len(c2_pts):
                    p1, p2 = c1_pts[j], c2_pts[i]

        # Added Path Length (rough, by consecutive segments in c2 that exceed thr)
        apl_mask = c2_d > thr if c2_d.size else np.array([False] * len(c2_pts))
        apl_idx = np.where(apl_mask)[0]
        apl_len = 0.0
        if apl_idx.size > 1:
            seg_lengths = []
            current = [c2_pts[apl_idx[0]]]
            for k in range(1, len(apl_idx)):
                if apl_idx[k] == apl_idx[k - 1] + 1:
                    current.append(c2_pts[apl_idx[k]])
                else:
                    if len(current) > 1:
                        seg_lengths.append(
                            float(np.linalg.norm(np.diff(np.array(current), axis=0), axis=1).sum())
                        )
                    current = [c2_pts[apl_idx[k]]]
            if len(current) > 1:
                seg_lengths.append(
                    float(np.linalg.norm(np.diff(np.array(current), axis=0), axis=1).sum())
                )
            apl_len = float(sum(seg_lengths))

        self.metrics = dict(
            dice_coefficient=dice_geom,
            jaccard_index=jaccard_geom,
            surface_dice=surf_dice,
            mean_surface_distance=msd,
            hausdorff_95=hd95,
            max_hausdorff=hd_max,
            volume_ratio=vol_ratio_geom,
            intersection_area=inter_a_geom,
            area1=a1_geom,
            area2=a2_geom,
            center_distance=centre_d,
            c1_points=c1_pts,
            c2_points=c2_pts,
            c1_min_dist=c1_d,
            c2_min_dist=c2_d,
            threshold=thr,
            percentile=perc,
            max_point_1=p1,
            max_point_2=p2,
            apl_mask=apl_mask,
            apl_length=apl_len,
        )
        return self.metrics

    # ------------------------------- PLOTS ----------------------------------
    def plot_surface_acceptability(self, ax, params):
        ax.set_title('Surface DICE @ Threshold', fontweight='bold')
        thr = self.metrics.get('threshold', params['distance_threshold'])
        c1_pts = self.metrics.get('c1_points')
        c2_pts = self.metrics.get('c2_points')
        c2_min_dist = self.metrics.get('c2_min_dist')
    
        c2_ok = np.array([False] * len(c2_pts if c2_pts is not None else []))
        if (c2_min_dist is not None and isinstance(c2_min_dist, np.ndarray) and c2_min_dist.size > 0 and
            c2_pts is not None and len(c2_min_dist) == len(c2_pts)):
            c2_ok = c2_min_dist <= thr
    
        c1_center = (params['circle1_x'], params['circle1_y'])
        r1 = params['radius1']
        r_inner, r_outer = max(r1 - thr, 0), r1 + thr
    
        # tolerance band around reference
        if r1 > 0:
            ax.add_patch(Circle(c1_center, r_outer, facecolor='lightgreen', alpha=0.25, edgecolor=None, zorder=0,
                                label=f'Ref. Tol. Band (±{thr:.1f}mm)'))
            ax.add_patch(Circle(c1_center, r_inner, facecolor='white', alpha=1.00, edgecolor=None, zorder=1))
    
        # reference surface
        if c1_pts is not None and len(c1_pts) > 0:
            ax.plot(np.append(c1_pts[:, 0], c1_pts[0, 0]), np.append(c1_pts[:, 1], c1_pts[0, 1]),
                    'blue', lw=1, label='Reference Surface', zorder=2)
    
        # test points classified by threshold
        if c2_pts is not None and len(c2_pts) > 0 and len(c2_ok) == len(c2_pts):
            ax.scatter(c2_pts[c2_ok, 0],  c2_pts[c2_ok, 1],  c='green', s=15, alpha=0.8, label='Test (Within Tol.)', zorder=3)
            ax.scatter(c2_pts[~c2_ok, 0], c2_pts[~c2_ok, 1], c='red',   s=20, alpha=0.9, label='Test (Outside Tol.)', zorder=3)
            ax.plot(np.append(c2_pts[:,0], c2_pts[0,0]), np.append(c2_pts[:,1], c2_pts[0,1]),
                    color='red', linestyle='--', lw=0.8, alpha=0.7, zorder=2, label='Test Surface Outline')
        elif c2_pts is not None and len(c2_pts) > 0:
            ax.scatter(c2_pts[:, 0], c2_pts[:, 1], c='gray', s=15, alpha=0.8, label='Test (accept. error)', zorder=3)
    
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')

    def plot_threshold_visualization(self, ax, params):
        ax.set_title("Surface Distance Analysis", fontweight="bold")
        c1_center = (params["circle1_x"], params["circle1_y"])
        c2_center = (params["circle2_x"], params["circle2_y"])
        r1, r2 = params["radius1"], params["radius2"]
        c1_pts, c2_pts = self.metrics["c1_points"], self.metrics["c2_points"]
        c1_d, c2_d = self.metrics["c1_min_dist"], self.metrics["c2_min_dist"]
        thr = self.metrics["threshold"]

        if params["noise1"] > 0 and len(c1_pts) > 0:
            ax.plot(
                np.append(c1_pts[:, 0], c1_pts[0, 0]),
                np.append(c1_pts[:, 1], c1_pts[0, 1]),
                "b-",
                lw=1,
                label="Reference Contour",
            )
        elif r1 > 0:
            ax.add_patch(Circle(c1_center, r1, fill=False, edgecolor="blue", lw=1, label="Reference Contour"))

        if params["noise2"] > 0 and len(c2_pts) > 0:
            ax.plot(
                np.append(c2_pts[:, 0], c2_pts[0, 0]),
                np.append(c2_pts[:, 1], c2_pts[0, 1]),
                "r-",
                lw=1,
                label="Test Contour",
            )
        elif r2 > 0:
            ax.add_patch(Circle(c2_center, r2, fill=False, edgecolor="red", lw=1, label="Test Contour"))

        if len(c1_pts) > 0 and c1_d.size == len(c1_pts):
            c1_ok = c1_d <= thr
            ax.scatter(c1_pts[c1_ok, 0], c1_pts[c1_ok, 1], c="green", s=15, alpha=0.7, label=f"Within {thr:.1f} mm")
            ax.scatter(c1_pts[~c1_ok, 0], c1_pts[~c1_ok, 1], c="orange", s=15, alpha=0.7, label=f"Beyond {thr:.1f} mm")
        if len(c2_pts) > 0 and c2_d.size == len(c2_pts):
            c2_ok = c2_d <= thr
            ax.scatter(c2_pts[c2_ok, 0], c2_pts[c2_ok, 1], c="green", s=15, alpha=0.7)
            ax.scatter(c2_pts[~c2_ok, 0], c2_pts[~c2_ok, 1], c="orange", s=15, alpha=0.7)

        if self.metrics.get("max_hausdorff", 0) > 0:
            p1_plot, p2_plot = self.metrics["max_point_1"], self.metrics["max_point_2"]
            if (
                isinstance(p1_plot, np.ndarray)
                and p1_plot.shape == (2,)
                and isinstance(p2_plot, np.ndarray)
                and p2_plot.shape == (2,)
                and not (np.all(p1_plot == 0) and np.all(p2_plot == 0))
            ):
                ax.plot(
                    [p1_plot[0], p2_plot[0]],
                    [p1_plot[1], p2_plot[1]],
                    "k--",
                    lw=1.5,
                    alpha=0.8,
                    label=f"Max Dist: {self.metrics['max_hausdorff']:.2f} mm",
                )
                ax.scatter([p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]], c="black", s=40, marker="X", alpha=0.9)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    def plot_distance_distribution(self, ax):
        ax.set_title("Surface Distance Distribution", fontweight="bold")
        c1_min_dist, c2_min_dist = self.metrics.get("c1_min_dist"), self.metrics.get("c2_min_dist")

        valid = []
        if isinstance(c1_min_dist, np.ndarray) and c1_min_dist.ndim == 1:
            f1 = c1_min_dist[np.isfinite(c1_min_dist)]
            if f1.size > 0:
                valid.append(f1)
        if isinstance(c2_min_dist, np.ndarray) and c2_min_dist.ndim == 1:
            f2 = c2_min_dist[np.isfinite(c2_min_dist)]
            if f2.size > 0:
                valid.append(f2)

        if not valid:
            ax.text(0.5, 0.5, "Not enough finite data for histogram", ha="center", va="center", transform=ax.transAxes)
        else:
            all_d = np.concatenate(valid)
            if all_d.size == 0:
                ax.text(0.5, 0.5, "No finite distances to plot", ha="center", va="center", transform=ax.transAxes)
            else:
                max_dist = float(np.max(all_d)) if all_d.size > 0 else 1.0
                bins = np.linspace(0, max(1.0, max_dist), 30)
                ax.hist(all_d, bins=bins, alpha=0.7, color="skyblue", edgecolor="black")

                msd = self.metrics.get("mean_surface_distance", np.nan)
                hd95 = self.metrics.get("hausdorff_95", np.nan)
                hdmax = self.metrics.get("max_hausdorff", np.nan)
                thr = self.metrics.get("threshold", np.nan)
                if np.isfinite(msd) and msd > 0:
                    ax.axvline(msd, color="red", linestyle="--", label=f"Mean: {msd:.2f}mm")
                if np.isfinite(hd95) and hd95 > 0:
                    ax.axvline(hd95, color="orange", linestyle="--", label=f"HD95: {hd95:.2f}mm")
                if np.isfinite(hdmax) and hdmax > 0:
                    ax.axvline(hdmax, color="purple", linestyle="--", label=f"Max: {hdmax:.2f}mm")
                if np.isfinite(thr) and thr > 0:
                    ax.axvline(thr, color="green", linestyle="--", label=f"Threshold: {thr:.2f}mm")
                ax.legend(fontsize=8)

        ax.set_xlabel("Distance (mm)")
        ax.set_ylabel("Frequency")
        ax.grid(True, alpha=0.3)

    def plot_overlap_regions(self, ax, params):
        ax.set_title("Volume Overlap (DICE Analysis)", fontweight="bold")
        c1_center = (params["circle1_x"], params["circle1_y"])
        c2_center = (params["circle2_x"], params["circle2_y"])
        r1, r2 = params["radius1"], params["radius2"]
        noise1, noise2 = params["noise1"], params["noise2"]
        c1_pts, c2_pts = self.metrics["c1_points"], self.metrics["c2_points"]

        if noise1 > 0 and len(c1_pts) > 0:
            ax.add_patch(Polygon(c1_pts, closed=True, facecolor="blue", alpha=0.30, edgecolor="blue", lw=1.0, label="Reference"))
        elif r1 > 0:
            ax.add_patch(Circle(c1_center, r1, facecolor="blue", alpha=0.30, edgecolor="blue", lw=1.0, label="Reference"))

        if noise2 > 0 and len(c2_pts) > 0:
            ax.add_patch(Polygon(c2_pts, closed=True, facecolor="red", alpha=0.30, edgecolor="red", lw=1.0, label="Test"))
        elif r2 > 0:
            ax.add_patch(Circle(c2_center, r2, facecolor="red", alpha=0.30, edgecolor="red", lw=1.0, label="Test"))

        if r1 > 0:
            ax.plot(*c1_center, "bo", ms=6, label="Ref Center")
        if r2 > 0:
            ax.plot(*c2_center, "ro", ms=6, label="Test Center")
        if r1 > 0 and r2 > 0:
            ax.plot([c1_center[0], c2_center[0]], [c1_center[1], c2_center[1]], "k--", alpha=0.5)

        ax.text(
            0.02,
            0.98,
            f'DICE (Geometric): {self.metrics.get("dice_coefficient",0):.3f}\nJaccard (Geometric): {self.metrics.get("jaccard_index",0):.3f}',
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    def plot_distance_heatmap(self, ax, fig_ref):
        ax.set_title("Distance Field Analysis", fontweight="bold")
        c1_pts, c2_pts = self.metrics.get("c1_points"), self.metrics.get("c2_points")
        c1_d, c2_d = self.metrics.get("c1_min_dist"), self.metrics.get("c2_min_dist")

        scat1 = None
        if len(c1_pts) > 0 and c1_d is not None and len(c1_d) == len(c1_pts):
            mask = np.isfinite(c1_d)
            if np.any(mask):
                scat1 = ax.scatter(c1_pts[mask, 0], c1_pts[mask, 1], c=c1_d[mask], cmap="viridis", vmin=0, s=20, alpha=0.7)
        mappable_for_cbar = scat1

        if len(c2_pts) > 0 and c2_d is not None and len(c2_d) == len(c2_pts):
            mask = np.isfinite(c2_d)
            if np.any(mask):
                current_scat = ax.scatter(c2_pts[mask, 0], c2_pts[mask, 1], c=c2_d[mask], cmap="viridis", vmin=0, s=20, alpha=0.7, marker="s")
                if mappable_for_cbar is None:
                    mappable_for_cbar = current_scat

        if mappable_for_cbar is not None:
            cax = inset_axes(ax, width="5%", height="70%", loc="center left",
                             bbox_to_anchor=(1.05, 0.0, 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            cbar = fig_ref.colorbar(mappable_for_cbar, cax=cax)
            cbar.set_label("Distance (mm)", rotation=270, labelpad=15)
        else:
            ax.text(0.5, 0.5, "Not enough finite data for heatmap", ha="center", va="center", transform=ax.transAxes)

        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)

    def plot_added_path_length(self, ax):
        ax.set_title("Added Path Length", fontweight="bold")
        c1_pts = self.metrics.get("c1_points")
        c2_pts = self.metrics.get("c2_points")
        mask = self.metrics.get("apl_mask")

        if c1_pts is not None and len(c1_pts) > 0:
            ax.plot(
                np.append(c1_pts[:, 0], c1_pts[0, 0]),
                np.append(c1_pts[:, 1], c1_pts[0, 1]),
                "blue",
                lw=1,
                label="Reference",
            )

        if c2_pts is not None and len(c2_pts) > 0 and mask is not None and len(mask) == len(c2_pts):
            ax.scatter(c2_pts[~mask, 0], c2_pts[~mask, 1], c="green", s=15, alpha=0.7, label="Accepted (in tolerance)")
            ax.scatter(c2_pts[mask, 0], c2_pts[mask, 1], c="red", s=20, alpha=0.9, label="Added Path Length (Needs Edit)")
        elif c2_pts is not None and len(c2_pts) > 0:
            ax.scatter(c2_pts[:, 0], c2_pts[:, 1], c="gray", s=15, alpha=0.7, label="Test (mask error)")

        ax.text(
            0.02,
            0.98,
            f"APL = {self.metrics.get('apl_length', 0):.2f} mm",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8),
        )
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    def get_metrics_text(self):
        return f"""
VOLUMETRIC OVERLAP METRICS (Geometric):
──────────────────────────────────────────────────────────────────
DICE Coefficient:           {self.metrics.get('dice_coefficient', np.nan):.4f}  (0-1, higher=better overlap)
Jaccard Index:              {self.metrics.get('jaccard_index', np.nan):.4f}  (0-1, alternative overlap metric)
Volume Ratio:               {self.metrics.get('volume_ratio', np.nan):.4f}  (size similarity)

SURFACE-BASED METRICS (Sampled Points):
──────────────────────────────────────────────────────────────────
Surface DICE:               {self.metrics.get('surface_dice', np.nan):.4f}  (surface point agreement @ threshold)
Mean Surface Distance:      {self.metrics.get('mean_surface_distance', np.nan):.3f} mm  (average error)
95th Percentile HD:         {self.metrics.get('hausdorff_95', np.nan):.3f} mm  (HD95, outlier-robust)
Maximum Hausdorff:          {self.metrics.get('max_hausdorff', np.nan):.3f} mm  (worst-case error)

GEOMETRIC PROPERTIES:
──────────────────────────────────────────────────────────────────
Reference Geo. Area:        {self.metrics.get('area1', np.nan):.2f} mm²
Test Geo. Area:             {self.metrics.get('area2', np.nan):.2f} mm²
Geo. Intersection Area:     {self.metrics.get('intersection_area', np.nan):.2f} mm²
Center-to-Center Distance:  {self.metrics.get('center_distance', np.nan):.3f} mm
Added Path Length (APL):    {self.metrics.get('apl_length', np.nan):.2f} mm (length of test contour outside threshold)

CLINICAL INTERPRETATION GUIDELINES (General):
──────────────────────────────────────────────────────────────────
DICE > 0.8:     Excellent agreement
DICE 0.7-0.8:   Good agreement
DICE 0.5-0.7:   Moderate agreement
DICE < 0.5:     Poor agreement

HD95 < 2mm:     Excellent precision (Context-dependent)
HD95 2-5mm:     Good precision (Context-dependent)
HD95 > 5mm:     Poor precision (Context-dependent)
        """.strip()


# =============================== STREAMLIT APP ================================

st.set_page_config(layout="wide", page_title="RadOnc Contour Metrics")
st.title("RadOnc Contour Overlap Metrics - Educational Tool 🩺🔬")
st.caption("Use the controls in the sidebar. Computations run when you click **Apply**.")

# deterministic RNG for noise (prevents flicker)
if "rng" not in st.session_state:
    st.session_state.rng = np.random.default_rng(12345)

# default params
if "params" not in st.session_state:
    st.session_state.params = {
        "circle1_x": 0.0,
        "circle1_y": 0.0,
        "radius1": 3.0,
        "noise1": 0.0,
        "circle2_x": 2.0,
        "circle2_y": 1.0,
        "radius2": 3.2,
        "noise2": 0.0,
        "distance_threshold": 1.0,
        "percentile": 95.0,
        "num_points": 200,
    }

if "noise_cache" not in st.session_state:
    st.session_state.noise_cache = defaultdict(dict)

# -------------------------- SIDEBAR: Controls (Debounced) --------------------
with st.sidebar.form("controls", clear_on_submit=False):
    st.subheader("Contour Parameters")
    with st.expander("Reference Contour (Blue)", expanded=True):
        st.session_state.params["circle1_x"] = st.slider("X Position (C1)", -10.0, 10.0, st.session_state.params["circle1_x"], 0.1, key="c1x")
        st.session_state.params["circle1_y"] = st.slider("Y Position (C1)", -10.0, 10.0, st.session_state.params["circle1_y"], 0.1, key="c1y")
        st.session_state.params["radius1"]   = st.slider("Radius (C1)",    0.0, 8.0,  st.session_state.params["radius1"],   0.1, key="r1")
        st.session_state.params["noise1"]    = st.slider("Surface Noise (C1)", 0.0, 1.0, st.session_state.params["noise1"], 0.05, key="n1")

    with st.expander("Test Contour (Red)", expanded=True):
        st.session_state.params["circle2_x"] = st.slider("X Position (C2)", -10.0, 10.0, st.session_state.params["circle2_x"], 0.1, key="c2x")
        st.session_state.params["circle2_y"] = st.slider("Y Position (C2)", -10.0, 10.0, st.session_state.params["circle2_y"], 0.1, key="c2y")
        st.session_state.params["radius2"]   = st.slider("Radius (C2)",    0.0, 8.0,  st.session_state.params["radius2"],   0.1, key="r2")
        st.session_state.params["noise2"]    = st.slider("Surface Noise (C2)", 0.0, 1.0, st.session_state.params["noise2"], 0.05, key="n2")

    st.subheader("Analysis Parameters")
    st.session_state.params["distance_threshold"] = st.slider("Distance Threshold (mm)", 0.1, 5.0, st.session_state.params["distance_threshold"], 0.1, key="dt")
    st.session_state.params["percentile"]         = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, st.session_state.params["percentile"], 0.1, key="perc")
    st.session_state.params["num_points"]         = st.slider("Sample Points per Contour", 10, 500, st.session_state.params["num_points"], 10, key="npts")

    cols = st.columns(2)
    with cols[0]:
        apply = st.form_submit_button("Apply 🔁")
    with cols[1]:
        reseed = st.form_submit_button("Reseed noise 🎲")

# Reset + About
st.sidebar.markdown("---")
if st.sidebar.button("Reset to Default", key="reset_button"):
    st.session_state.params = {
        "circle1_x": 0.0, "circle1_y": 0.0, "radius1": 4.0, "noise1": 0.0,
        "circle2_x": 1.0, "circle2_y": 1.0, "radius2": 4.2, "noise2": 0.0,
        "distance_threshold": 1.0, "percentile": 95.0, "num_points": 200
    }
    st.session_state.noise_cache = defaultdict(dict)
    st.session_state.rng = np.random.default_rng(12345)
    st.experimental_rerun()

st.sidebar.markdown("---")
with st.sidebar.expander("Metric Definitions & Info", expanded=False):
    st.markdown("""
- **DICE Coefficient (Geometric)**: Volumetric overlap between perfect circles via geometry.
- **Jaccard Index (Geometric)**: Alternative volumetric overlap metric for perfect circles.
- **Volume Ratio (Geometric)**: Ratio of smaller to larger geometric circle area.
- **Surface DICE**: Agreement of sampled surface points based on distance threshold.
- **Mean Surface Distance (MSD)**: Average nearest-surface distance.
- **Hausdorff Distance (HD95)**: 95th percentile of nearest-surface distances.
- **Max Hausdorff**: Maximum nearest-surface distance.
- **Added Path Length (APL)**: Approximate length of the test contour outside tolerance.
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### About & License")
st.sidebar.markdown(f"""
**Developer:** Ciaran Malone  
**Version:** 1.2.0  

**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0)
""")

# Handle reseed
if "initialized" not in st.session_state:
    st.session_state.initialized = True
if reseed:
    # reinitialize RNG with a new random seed based on time
    st.session_state.rng = np.random.default_rng(int(time.time()) % (2**32 - 1))
    st.info("Noise reseeded. Click **Apply** to recompute.")

# Compute metrics (only when Apply pressed or on first load)
analyzer = st.session_state.get("analyzer") or ContourAnalysisApp()
if apply or "last_metrics" not in st.session_state:
    try:
        current_metrics = analyzer.compute_metrics(st.session_state.params, st.session_state.noise_cache)
    except Exception as e:
        st.error(f"Error computing metrics: {e}.")
        st.stop()
    st.session_state.last_metrics = current_metrics
    st.session_state.analyzer = analyzer
else:
    current_metrics = st.session_state.last_metrics

# ------------------------------ VISUALIZATIONS -------------------------------

st.header("Contour Analysis Visualizations")

tab1, tab2 = st.tabs(["Overview", "Distance/Heat"])

with tab1:
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    analyzer.plot_threshold_visualization(axes1[0], st.session_state.params)
    analyzer.plot_surface_acceptability(axes1[1], st.session_state.params)
    analyzer.plot_overlap_regions(axes1[2], st.session_state.params)
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=True)

with tab2:
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    analyzer.plot_distance_distribution(axes2[0])
    analyzer.plot_distance_heatmap(axes2[1], fig2)
    analyzer.plot_added_path_length(axes2[2])
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=True)

# --------------- LAZY PNG EXPORT (build only when requested) -----------------

st.sidebar.markdown("---")
st.sidebar.subheader("Export")
if st.sidebar.button("Prepare PNG 📷"):
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        analyzer.plot_threshold_visualization(axes[0, 0], st.session_state.params)
        analyzer.plot_distance_distribution(axes[0, 1])
        analyzer.plot_surface_acceptability(axes[0, 2], st.session_state.params)
        analyzer.plot_overlap_regions(axes[1, 0], st.session_state.params)
        analyzer.plot_distance_heatmap(axes[1, 1], fig)
        analyzer.plot_added_path_length(axes[1, 2])
        fig.tight_layout(rect=[0, 0.03, 1, 0.98])

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        buf.seek(0)
        st.session_state.png_bytes = buf.getvalue()
        st.success("PNG prepared. Click download below.")
    except Exception as e:
        st.error(f"Failed to prepare PNG: {e}")

if "png_bytes" in st.session_state:
    st.sidebar.download_button(
        label="💾 Download Plot as PNG",
        data=st.session_state.png_bytes,
        file_name=f"contour_analysis_{int(time.time())}.png",
        mime="image/png",
    )

# ------------------------------- METRICS TEXT --------------------------------

st.header("Computed Metrics")
metrics_str = analyzer.get_metrics_text()
st.text_area("Metrics Summary", metrics_str, height=480, key="metrics_summary_area")

st.sidebar.download_button(
    label="📄 Export Metrics as TXT",
    data=metrics_str,
    file_name=f"metrics_{int(time.time())}.txt",
    mime="text/plain",
    key="export_metrics_btn",
)

