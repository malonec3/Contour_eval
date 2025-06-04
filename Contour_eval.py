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

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.spatial.distance
import scipy.interpolate
from scipy.interpolate import CubicSpline
from collections import defaultdict
import time
import io

# --- Core Logic and Plotting Class ---
class ContourAnalysisApp:
    def __init__(self):
        self.metrics = {} # Stores computed metrics

    def circle_intersection_area(self, r1, r2, d):
        """Calculate intersection area of two circles with robust handling."""
        if d >= r1 + r2:
            return 0  # No intersection
        if d <= abs(r1 - r2): # Handles one circle inside another, or identical circles if d=0
            return np.pi * min(r1, r2)**2
        if d == 0 and r1 == r2: # Already covered by d <= abs(r1-r2) if r1=r2, but explicit doesn't harm
            return np.pi * r1**2

        # If d=0 and radii are different, it's caught by d <= abs(r1-r2)
        # If r1 or r2 is 0 and d > 0, it's caught by d >= r1+r2 (no intersection if d >= other_radius)
        # or d <= abs(r1-r2) (e.g. r1=0, d < r2 means point is inside circle, area is 0 for point).
        # The min(r1,r2)**2 in d <= abs(r1-r2) handles case where one radius is 0.

        try:
            r1_sq, r2_sq, d_sq = r1**2, r2**2, d**2
            # Ensure denominators are not zero if d or radii are zero
            # This should be prevented by earlier checks, but as a safeguard for arccos inputs:
            if d == 0 or r1 == 0 or r2 == 0: # Should have been caught, implies unusual state
                 # This scenario implies d > 0, one radius non-zero, other zero.
                 # If r1=0, cos_angle1 is ill-defined. Intersection must be 0.
                 # If r2=0, cos_angle2 is ill-defined. Intersection must be 0.
                 # This path should ideally not be reached if prior logic is perfect.
                 return 0


            cos_angle1 = (d_sq + r1_sq - r2_sq) / (2 * d * r1)
            cos_angle2 = (d_sq + r2_sq - r1_sq) / (2 * d * r2)
            
            cos_angle1 = np.clip(cos_angle1, -1, 1)
            cos_angle2 = np.clip(cos_angle2, -1, 1)
            
            angle1 = 2 * np.arccos(cos_angle1)
            angle2 = 2 * np.arccos(cos_angle2)
            
            area1 = 0.5 * r1_sq * (angle1 - np.sin(angle1))
            area2 = 0.5 * r2_sq * (angle2 - np.sin(angle2))
            
            return area1 + area2
        except (ValueError, ZeroDivisionError):
            # Fallback, though prior checks aim to prevent this.
            # This might occur if d is extremely small but not exactly zero, leading to instability.
            # Re-evaluating based on conditions might be safer here.
            if d >= r1 + r2: return 0
            if d <= abs(r1 - r2): return np.pi * min(r1, r2)**2
            return 0 # Default to no intersection if unexpected error

    def generate_circle_points(self, center, radius, num_points, noise_level, key, noise_cache):
        cache = noise_cache[key]
        regenerate = (cache.get('noise_level') != noise_level or
                      cache.get('num_points') != num_points or
                      cache.get('base_radius') != radius) # Add radius check for noise consistency if radius changes

        if regenerate:
            angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
            if noise_level <= 0.0:
                delta_frac = np.zeros(num_points)
                delta_theta = np.zeros(num_points)
            else:
                n_knots = max(8, int(20 * noise_level)) # Number of knots for spline
                knot_ang = np.linspace(0, 2 * np.pi, n_knots, endpoint=False)
                
                # Generate random perturbations for knots
                # Scale noise effect slightly by inverse of radius to make it less visually dominant on small circles
                # Or keep it absolute as it is: noise_level is an absolute measure. Current approach is fine.
                rad_knot = np.random.normal(0, 0.25 * noise_level, n_knots) # Radial noise
                th_knot = np.random.normal(0, 0.10 * noise_level, n_knots)  # Angular noise (tangential)

                # Ensure periodic boundary conditions for smooth closure
                knot_ang_periodic = np.append(knot_ang, knot_ang[0] + 2 * np.pi)
                rad_knot_periodic = np.append(rad_knot, rad_knot[0])
                th_knot_periodic = np.append(th_knot, th_knot[0])

                cs_r = CubicSpline(knot_ang_periodic, rad_knot_periodic, bc_type='periodic')
                cs_th = CubicSpline(knot_ang_periodic, th_knot_periodic, bc_type='periodic')

                delta_frac = cs_r(angles)     # Radial fractional change r_new = r_base * (1 + delta_frac)
                delta_theta = cs_th(angles)   # Angular offset theta_new = theta_base + delta_theta
            
            cache.update({'noise_level': noise_level,
                          'num_points': num_points,
                          'base_radius': radius, # Cache base_radius for noise consistency
                          'delta_frac': delta_frac,
                          'delta_theta': delta_theta})

        angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        # Retrieve from cache
        delta_frac = cache['delta_frac']
        delta_theta = cache['delta_theta']
        
        # Apply perturbations
        # Note: delta_frac is fractional, delta_theta is absolute angle offset
        r_perturbed = radius * (1.0 + delta_frac) 
        # Ensure radius doesn't go negative if noise is extreme, though delta_frac is usually small
        r_perturbed = np.maximum(r_perturbed, 0) 
        th_perturbed = angles + delta_theta

        x = center[0] + r_perturbed * np.cos(th_perturbed)
        y = center[1] + r_perturbed * np.sin(th_perturbed)
        return np.column_stack([x, y])

    def compute_metrics(self, params, noise_cache):
        c1_center = (params['circle1_x'], params['circle1_y'])
        c2_center = (params['circle2_x'], params['circle2_y'])
        r1_param = params['radius1'] # Base radius for c1
        r2_param = params['radius2'] # Base radius for c2
        thr = params['distance_threshold']
        perc = params['percentile']
        n_pts = params['num_points']

        # Generate points based on base radius and noise
        c1_pts = self.generate_circle_points(c1_center, r1_param, n_pts,
                                             params['noise1'], key='c1', noise_cache=noise_cache)
        c2_pts = self.generate_circle_points(c2_center, r2_param, n_pts,
                                             params['noise2'], key='c2', noise_cache=noise_cache)

        # Volumetric metrics are based on the *intended* perfect circle radii for this tool
        # unless specified otherwise. If noise is applied, the "actual" area/volume would
        # be from the polygon, but traditionally DICE on circles uses geometric formula.
        # Current implementation uses geometric formula for circle areas/intersection.
        # Surface metrics use the generated (possibly noisy) points.
        a1_geom = np.pi * r1_param**2 # Geometric area of circle 1
        a2_geom = np.pi * r2_param**2 # Geometric area of circle 2

        centre_d = np.linalg.norm(np.array(c1_center) - np.array(c2_center))
        inter_a_geom = self.circle_intersection_area(r1_param, r2_param, centre_d)
        union_a_geom = a1_geom + a2_geom - inter_a_geom

        dice_geom = 2 * inter_a_geom / (a1_geom + a2_geom) if (a1_geom + a2_geom) > 0 else 0
        jaccard_geom = inter_a_geom / union_a_geom if union_a_geom > 0 else 0
        vol_ratio_geom = min(a1_geom, a2_geom) / max(a1_geom, a2_geom) if max(a1_geom, a2_geom) > 0 else 0

        # Surface metrics
        d_mat = scipy.spatial.distance.cdist(c1_pts, c2_pts, 'euclidean') if n_pts > 0 else np.array([])
        
        c1_d = np.array([np.inf]) # Default for c1_d
        c2_d = np.array([np.inf]) # Default for c2_d

        if d_mat.size > 0:
            if d_mat.shape[1] > 0 : c1_d = np.min(d_mat, axis=1) 
            if d_mat.shape[0] > 0 : c2_d = np.min(d_mat, axis=0)
        elif n_pts > 0: # Points exist but no other contour to compare against
            # This case should mean one of the contours has 0 points if d_mat is empty.
            # But if n_pts > 0 for both, d_mat should not be empty.
            # If only one set of points exists (e.g. r2=0, noise2=0), d_mat is (N,0) or (0,N)
            # which scipy handles by returning empty array. Let's assume if cX_pts has points,
            # its distances to an empty set are infinite.
             if len(c1_pts) > 0 and len(c2_pts) == 0: c1_d = np.full(len(c1_pts), np.inf)
             if len(c2_pts) > 0 and len(c1_pts) == 0: c2_d = np.full(len(c2_pts), np.inf)

        
        surf_dice = 0
        if n_pts > 0 and (c1_d.size > 0 or c2_d.size > 0): # Ensure there are distances to sum
            num_valid_c1_d = (c1_d <= thr).sum() if c1_d.size > 0 and not np.all(np.isinf(c1_d)) else 0
            num_valid_c2_d = (c2_d <= thr).sum() if c2_d.size > 0 and not np.all(np.isinf(c2_d)) else 0
            surf_dice = (num_valid_c1_d + num_valid_c2_d) / (len(c1_pts) + len(c2_pts)) if (len(c1_pts) + len(c2_pts)) > 0 else 0


        msd = 0
        if c1_d.size > 0 and not np.all(np.isinf(c1_d)) and c2_d.size > 0 and not np.all(np.isinf(c2_d)):
            msd = (np.mean(c1_d[np.isfinite(c1_d)]) + np.mean(c2_d[np.isfinite(c2_d)])) / 2
        elif c1_d.size > 0 and not np.all(np.isinf(c1_d)):
            msd = np.mean(c1_d[np.isfinite(c1_d)])
        elif c2_d.size > 0 and not np.all(np.isinf(c2_d)):
            msd = np.mean(c2_d[np.isfinite(c2_d)])
        
        hd95 = 0
        valid_c1_d_finite = c1_d[np.isfinite(c1_d)]
        valid_c2_d_finite = c2_d[np.isfinite(c2_d)]
        if valid_c1_d_finite.size > 0 and valid_c2_d_finite.size > 0 :
            hd95 = max(np.percentile(valid_c1_d_finite, perc), np.percentile(valid_c2_d_finite, perc))
        elif valid_c1_d_finite.size > 0:
            hd95 = np.percentile(valid_c1_d_finite, perc)
        elif valid_c2_d_finite.size > 0:
            hd95 = np.percentile(valid_c2_d_finite, perc)
            
        hd_max = 0
        p1, p2 = np.array([0,0]), np.array([0,0]) 

        all_finite_distances = np.concatenate([valid_c1_d_finite, valid_c2_d_finite])
        if all_finite_distances.size > 0:
            hd_max = np.max(all_finite_distances)
            # Find points for hd_max. This part can be complex if hd_max comes from c1_d or c2_d
            if valid_c1_d_finite.size > 0 and (valid_c2_d_finite.size == 0 or np.max(valid_c1_d_finite) >= np.max(valid_c2_d_finite)):
                if np.max(valid_c1_d_finite) == hd_max: # Check if max is from c1_d
                    p1_idx_hd = np.argmax(c1_d) # Use original c1_d to find index
                    if p1_idx_hd < len(c1_pts) and d_mat.shape[1] > 0:
                         p1 = c1_pts[p1_idx_hd]
                         p2 = c2_pts[np.argmin(d_mat[p1_idx_hd])]
            elif valid_c2_d_finite.size > 0: # Max must be from c2_d
                 if np.max(valid_c2_d_finite) == hd_max:
                    p2_idx_hd = np.argmax(c2_d) # Use original c2_d
                    if p2_idx_hd < len(c2_pts) and d_mat.shape[0] > 0:
                        p2 = c2_pts[p2_idx_hd]
                        p1 = c1_pts[np.argmin(d_mat[:, p2_idx_hd])]
        
        apl_mask = c2_d > thr if c2_d.size > 0 else np.array([False]*len(c2_pts))
        apl_idx  = np.where(apl_mask)[0]
        apl_len = 0.0
        if apl_idx.size > 1:
            valid_apl_idx = apl_idx[apl_idx < len(c2_pts)]
            if valid_apl_idx.size > 1:
                 # Sum distances between consecutive points in the 'added path' segments
                 # This needs to handle disjoint segments correctly by summing lengths of segments
                 # The current method sums lengths between consecutive points in valid_apl_idx.
                 # If valid_apl_idx are [1,2,3, 7,8], it sums (d(1,2)+d(2,3)) + (d(3,7)) + (d(7,8)).
                 # This is not strictly APL if segments are disjoint.
                 # True APL would identify segments and sum their individual path lengths.
                 # For simplicity, current method is kept but noted as an approximation if segments are disjoint.
                 # A more accurate APL would involve finding connected components in apl_idx.
                 seg_lengths = []
                 if len(valid_apl_idx) > 0:
                    current_segment_pts = [c2_pts[valid_apl_idx[0]]]
                    for i in range(1, len(valid_apl_idx)):
                        # If points are consecutive in original contour, they are part of same APL segment
                        if valid_apl_idx[i] == valid_apl_idx[i-1] + 1:
                            current_segment_pts.append(c2_pts[valid_apl_idx[i]])
                        else: # Discontinuity, end of segment
                            if len(current_segment_pts) > 1:
                                seg_lengths.append(np.sum(np.linalg.norm(np.diff(np.array(current_segment_pts), axis=0), axis=1)))
                            current_segment_pts = [c2_pts[valid_apl_idx[i]]] # Start new segment
                    if len(current_segment_pts) > 1: # Last segment
                        seg_lengths.append(np.sum(np.linalg.norm(np.diff(np.array(current_segment_pts), axis=0), axis=1)))
                    apl_len = sum(seg_lengths)


        self.metrics = dict(
            dice_coefficient=dice_geom, jaccard_index=jaccard_geom, surface_dice=surf_dice,
            mean_surface_distance=msd, hausdorff_95=hd95, max_hausdorff=hd_max,
            volume_ratio=vol_ratio_geom, intersection_area=inter_a_geom, 
            area1=a1_geom, area2=a2_geom, # These are geometric areas
            center_distance=centre_d, c1_points=c1_pts, c2_points=c2_pts,
            c1_min_dist=c1_d, c2_min_dist=c2_d, distances=d_mat, threshold=thr,
            percentile=perc, max_point_1=p1, max_point_2=p2, apl_mask=apl_mask,
            apl_length=apl_len
        )
        return self.metrics

    def plot_threshold_visualization(self, ax, params):
        ax.set_title('Surface Distance Analysis', fontweight='bold')
        c1_center = (params['circle1_x'], params['circle1_y'])
        c2_center = (params['circle2_x'], params['circle2_y'])
        r1, r2 = params['radius1'], params['radius2']
        c1_pts, c2_pts = self.metrics['c1_points'], self.metrics['c2_points']
        c1_d, c2_d = self.metrics['c1_min_dist'], self.metrics['c2_min_dist']
        thr = self.metrics['threshold']

        if params['noise1'] > 0 and len(c1_pts) > 0:
            ax.plot(np.append(c1_pts[:, 0], c1_pts[0, 0]), np.append(c1_pts[:, 1], c1_pts[0, 1]),
                    'b-', lw=1, label='Reference Contour')
        elif r1 > 0 : 
            ax.add_patch(Circle(c1_center, r1, fill=False, edgecolor='blue', lw=1, label='Reference Contour'))

        if params['noise2'] > 0 and len(c2_pts) > 0:
            ax.plot(np.append(c2_pts[:, 0], c2_pts[0, 0]), np.append(c2_pts[:, 1], c2_pts[0, 1]),
                    'r-', lw=1, label='Test Contour')
        elif r2 > 0: 
            ax.add_patch(Circle(c2_center, r2, fill=False, edgecolor='red', lw=1, label='Test Contour'))
        
        if len(c1_pts)>0 and c1_d.size == len(c1_pts):
            c1_ok = c1_d <= thr
            ax.scatter(c1_pts[c1_ok, 0], c1_pts[c1_ok, 1], c='green', s=15, alpha=0.7, label=f'Within {thr:.1f} mm')
            ax.scatter(c1_pts[~c1_ok, 0], c1_pts[~c1_ok, 1], c='orange', s=15, alpha=0.7, label=f'Beyond {thr:.1f} mm')
        if len(c2_pts)>0 and c2_d.size == len(c2_pts):
            c2_ok = c2_d <= thr
            ax.scatter(c2_pts[c2_ok, 0], c2_pts[c2_ok, 1], c='green', s=15, alpha=0.7)
            ax.scatter(c2_pts[~c2_ok, 0], c2_pts[~c2_ok, 1], c='orange', s=15, alpha=0.7)

        if 'max_point_1' in self.metrics and self.metrics.get('max_hausdorff', 0) > 0: 
            p1_plot, p2_plot = self.metrics['max_point_1'], self.metrics['max_point_2']
            # Ensure points are valid 2D coordinates before plotting
            if isinstance(p1_plot, np.ndarray) and p1_plot.shape == (2,) and \
               isinstance(p2_plot, np.ndarray) and p2_plot.shape == (2,) and \
               not (np.all(p1_plot == 0) and np.all(p2_plot == 0)): # Avoid plotting default 0,0 if no real points found
                ax.plot([p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]], 'k--', lw=1.5, alpha=0.8,
                        label=f'Max Dist: {self.metrics["max_hausdorff"]:.2f} mm')
                ax.scatter([p1_plot[0], p2_plot[0]], [p1_plot[1], p2_plot[1]], c='black', s=40, marker='X', alpha=0.9)

        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # Remove duplicate labels in legend
        if by_label: ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')


    def plot_distance_distribution(self, ax):
        ax.set_title('Surface Distance Distribution', fontweight='bold')
        c1_min_dist, c2_min_dist = self.metrics.get('c1_min_dist'), self.metrics.get('c2_min_dist')
        
        valid_distances = []
        if c1_min_dist is not None and isinstance(c1_min_dist, np.ndarray) and c1_min_dist.ndim == 1:
            finite_c1_dist = c1_min_dist[np.isfinite(c1_min_dist)]
            if finite_c1_dist.size > 0: valid_distances.append(finite_c1_dist)
        if c2_min_dist is not None and isinstance(c2_min_dist, np.ndarray) and c2_min_dist.ndim == 1:
            finite_c2_dist = c2_min_dist[np.isfinite(c2_min_dist)]
            if finite_c2_dist.size > 0: valid_distances.append(finite_c2_dist)


        if not valid_distances:
            ax.text(0.5, 0.5, "Not enough finite data for histogram", ha='center', va='center', transform=ax.transAxes)
        else:
            all_distances = np.concatenate(valid_distances)
            if all_distances.size == 0:
                 ax.text(0.5, 0.5, "No finite distances to plot", ha='center', va='center', transform=ax.transAxes)
            else:
                max_dist = np.max(all_distances) if all_distances.size > 0 else 1.0
                bins = np.linspace(0, max_dist if max_dist > 0 else 1.0, 30) 
                ax.hist(all_distances, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
                
                if self.metrics.get('mean_surface_distance', 0) > 0 or not np.isnan(self.metrics.get('mean_surface_distance', np.nan)):
                    ax.axvline(self.metrics['mean_surface_distance'], color='red', linestyle='--', label=f'Mean: {self.metrics["mean_surface_distance"]:.2f}mm')
                if self.metrics.get('hausdorff_95', 0) > 0 or not np.isnan(self.metrics.get('hausdorff_95', np.nan)):
                    ax.axvline(self.metrics['hausdorff_95'], color='orange', linestyle='--', label=f'HD95: {self.metrics["hausdorff_95"]:.2f}mm')
                if self.metrics.get('max_hausdorff', 0) > 0 or not np.isnan(self.metrics.get('max_hausdorff', np.nan)):
                    ax.axvline(self.metrics['max_hausdorff'], color='purple', linestyle='--', label=f'Max: {self.metrics["max_hausdorff"]:.2f}mm')
                ax.axvline(self.metrics['threshold'], color='green', linestyle='--', label=f'Threshold: {self.metrics["threshold"]:.2f}mm')
                ax.legend(fontsize=8)
        
        ax.set_xlabel('Distance (mm)'); ax.set_ylabel('Frequency')
        ax.grid(True, alpha=0.3)

    def plot_overlap_regions(self, ax, params):
        ax.set_title('Volume Overlap (DICE Analysis)', fontweight='bold')
        c1_center = (params['circle1_x'], params['circle1_y'])
        c2_center = (params['circle2_x'], params['circle2_y'])
        r1, r2 = params['radius1'], params['radius2']
        noise1, noise2 = params['noise1'], params['noise2']
        c1_pts, c2_pts = self.metrics['c1_points'], self.metrics['c2_points']

        # Plot based on geometric radius if no noise, or actual points if noise
        if noise1 > 0 and len(c1_pts) > 0:
            ax.add_patch(Polygon(c1_pts, closed=True, facecolor='blue', alpha=0.30, edgecolor='blue', lw=1.0, label='Reference'))
        elif r1 > 0: # Perfect circle
            ax.add_patch(Circle(c1_center, r1, facecolor='blue', alpha=0.30, edgecolor='blue', lw=1.0, label='Reference'))

        if noise2 > 0 and len(c2_pts) > 0:
            ax.add_patch(Polygon(c2_pts, closed=True, facecolor='red', alpha=0.30, edgecolor='red', lw=1.0, label='Test'))
        elif r2 > 0: # Perfect circle
            ax.add_patch(Circle(c2_center, r2, facecolor='red', alpha=0.30, edgecolor='red', lw=1.0, label='Test'))

        if r1>0: ax.plot(*c1_center, 'bo', ms=6, label='Ref Center') # Smaller marker size
        if r2>0: ax.plot(*c2_center, 'ro', ms=6, label='Test Center') # Smaller marker size
        if r1>0 and r2>0: ax.plot([c1_center[0], c2_center[0]], [c1_center[1], c2_center[1]], 'k--', alpha=0.5)

        ax.text(0.02, 0.98, f'DICE (Geometric): {self.metrics.get("dice_coefficient",0):.3f}\nJaccard (Geometric): {self.metrics.get("jaccard_index",0):.3f}',
                transform=ax.transAxes, va='top', fontsize=8, bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')


    def plot_distance_heatmap(self, ax, fig_ref): 
        ax.set_title('Distance Field Analysis', fontweight='bold')
        c1_pts, c2_pts = self.metrics.get('c1_points'), self.metrics.get('c2_points')
        c1_d, c2_d = self.metrics.get('c1_min_dist'), self.metrics.get('c2_min_dist')
        
        scat1 = None
        # Ensure distances are finite for coloring
        if c1_pts is not None and len(c1_pts) > 0 and c1_d is not None and len(c1_d) == len(c1_pts):
            finite_c1_d_mask = np.isfinite(c1_d)
            if np.any(finite_c1_d_mask):
                 scat1 = ax.scatter(c1_pts[finite_c1_d_mask, 0], c1_pts[finite_c1_d_mask, 1], 
                                    c=c1_d[finite_c1_d_mask], cmap='viridis', vmin=0, s=20, alpha=0.7)
        
        if c2_pts is not None and len(c2_pts) > 0 and c2_d is not None and len(c2_d) == len(c2_pts):
            finite_c2_d_mask = np.isfinite(c2_d)
            if np.any(finite_c2_d_mask):
                 # Use scat1 for colorbar if it exists, otherwise try to create a mappable from this scatter
                 mappable_for_cbar = scat1
                 current_scat = ax.scatter(c2_pts[finite_c2_d_mask, 0], c2_pts[finite_c2_d_mask, 1], 
                               c=c2_d[finite_c2_d_mask], cmap='viridis', vmin=0, s=20, alpha=0.7, marker='s')
                 if not mappable_for_cbar: mappable_for_cbar = current_scat


        if 'mappable_for_cbar' in locals() and mappable_for_cbar: 
            cax = inset_axes(ax, width="5%", height="70%", loc='center left', 
                             bbox_to_anchor=(1.05, 0., 1, 1), bbox_transform=ax.transAxes, borderpad=0)
            cbar = fig_ref.colorbar(mappable_for_cbar, cax=cax)
            cbar.set_label('Distance (mm)', rotation=270, labelpad=15)
        else:
            ax.text(0.5, 0.5, "Not enough finite data for heatmap", ha='center', va='center', transform=ax.transAxes)

        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)

    def plot_added_path_length(self, ax):
        ax.set_title('Added Path Length', fontweight='bold')
        c1_pts = self.metrics.get('c1_points')
        c2_pts = self.metrics.get('c2_points')
        mask = self.metrics.get('apl_mask') # These are indices on c2_pts that are > threshold

        if c1_pts is not None and len(c1_pts)>0:
            ax.plot(np.append(c1_pts[:, 0], c1_pts[0, 0]), np.append(c1_pts[:, 1], c1_pts[0, 1]),
                    'blue', lw=1, label='Reference') # Changed to blue for consistency with other plots

        if c2_pts is not None and len(c2_pts)>0 and mask is not None and len(mask) == len(c2_pts): 
            ax.scatter(c2_pts[~mask, 0], c2_pts[~mask, 1], c='green', s=15, alpha=0.7, label='Test (Accepted)')
            ax.scatter(c2_pts[mask, 0], c2_pts[mask, 1], c='red', s=20, alpha=0.9, label='Test (Needs Edit - APL)')
        elif c2_pts is not None and len(c2_pts)>0: 
            ax.scatter(c2_pts[:, 0], c2_pts[:, 1], c='gray', s=15, alpha=0.7, label='Test (mask error)')

        ax.text(0.02, 0.98, f'APL = {self.metrics.get("apl_length", 0):.2f} mm', 
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', fc='white', alpha=0.8))
        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')


    def plot_surface_acceptability(self, ax, params):
        ax.set_title('Surface DICE @ Threshold', fontweight='bold')
        thr = self.metrics.get('threshold', params['distance_threshold'])
        c1_pts = self.metrics.get('c1_points')
        c2_pts = self.metrics.get('c2_points')
        c2_min_dist = self.metrics.get('c2_min_dist')
        
        c2_ok = np.array([False] * len(c2_pts if c2_pts is not None else [])) # Default
        if c2_min_dist is not None and isinstance(c2_min_dist, np.ndarray) and c2_min_dist.size > 0 and \
           c2_pts is not None and len(c2_min_dist) == len(c2_pts):
             c2_ok = c2_min_dist <= thr
        
        c1_center = (params['circle1_x'], params['circle1_y'])
        r1 = params['radius1']
        r_inner, r_outer = max(r1 - thr, 0), r1 + thr

        if r1 > 0 : 
            ax.add_patch(Circle(c1_center, r_outer, facecolor='lightgreen', alpha=0.25, edgecolor=None, zorder=0, label=f'Ref. Tol. Band (±{thr:.1f}mm)'))
            ax.add_patch(Circle(c1_center, r_inner, facecolor='white', alpha=1.00, edgecolor=None, zorder=1)) # "erase" inner part
        
        if c1_pts is not None and len(c1_pts)>0:
            ax.plot(np.append(c1_pts[:, 0], c1_pts[0, 0]), np.append(c1_pts[:, 1], c1_pts[0, 1]),
                    'blue', lw=1, label='Reference Surface', zorder=2) # Changed to blue
        
        if c2_pts is not None and len(c2_pts)>0 and len(c2_ok) == len(c2_pts):
            ax.scatter(c2_pts[c2_ok, 0], c2_pts[c2_ok, 1], c='green', s=15, alpha=0.8, label='Test (Within Tol.)', zorder=3)
            ax.scatter(c2_pts[~c2_ok, 0], c2_pts[~c2_ok, 1], c='red', s=20, alpha=0.9, label='Test (Outside Tol.)', zorder=3)
            # Test contour outline
            ax.plot(np.append(c2_pts[:,0], c2_pts[0,0]), np.append(c2_pts[:,1], c2_pts[0,1]), 
                    color='red', linestyle='--', lw=0.8, alpha=0.7, zorder=2, label='Test Surface Outline')
        elif c2_pts is not None and len(c2_pts)>0: 
            ax.scatter(c2_pts[:, 0], c2_pts[:, 1], c='gray', s=15, alpha=0.8, label='Test (accept. error)', zorder=3)

        ax.set_xlim(-10, 10); ax.set_ylim(-10, 10)
        ax.set_aspect('equal'); ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)')
        ax.grid(True, alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        if by_label: ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc='upper right')


    def get_metrics_text(self):
        # Using .get(metric, np.nan) to show NaN if metric is missing, helps debugging
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

# --- Streamlit App ---
st.set_page_config(layout="wide", page_title="RadOnc Contour Metrics") # Updated page_title
st.title("RadOnc Contour Overlap Metrics - Educational Tool 🩺🔬") # Updated title
st.caption("Scroll down in the sidebar for Developer Info, License, and Metric Definitions.") # Added caption

if 'params' not in st.session_state:
    st.session_state.params = {
        'circle1_x': 0.0, 'circle1_y': 0.0, 'radius1': 3.0, 'noise1': 0.0,
        'circle2_x': 2.0, 'circle2_y': 1.0, 'radius2': 3.2, 'noise2': 0.0,
        'distance_threshold': 1.0, 'percentile': 95.0, 'num_points': 200
    }
if 'noise_cache' not in st.session_state:
    st.session_state.noise_cache = defaultdict(dict)

st.sidebar.header("Contour Parameters")
with st.sidebar.expander("Reference Contour (Blue)", expanded=True):
    st.session_state.params['circle1_x'] = st.slider("X Position (C1)", -10.0, 10.0, st.session_state.params['circle1_x'], 0.1, key="c1x", help="X-coordinate of the center of the reference contour.")
    st.session_state.params['circle1_y'] = st.slider("Y Position (C1)", -10.0, 10.0, st.session_state.params['circle1_y'], 0.1, key="c1y", help="Y-coordinate of the center of the reference contour.")
    st.session_state.params['radius1'] = st.slider("Radius (C1)", 0.0, 8.0, st.session_state.params['radius1'], 0.1, key="r1", help="Radius of the reference contour (if no noise).")
    st.session_state.params['noise1'] = st.slider("Surface Noise (C1)", 0.0, 1.0, st.session_state.params['noise1'], 0.05, key="n1", help="Magnitude of random noise applied to the reference contour surface.")

with st.sidebar.expander("Test Contour (Red)", expanded=True):
    st.session_state.params['circle2_x'] = st.slider("X Position (C2)", -10.0, 10.0, st.session_state.params['circle2_x'], 0.1, key="c2x", help="X-coordinate of the center of the test contour.")
    st.session_state.params['circle2_y'] = st.slider("Y Position (C2)", -10.0, 10.0, st.session_state.params['circle2_y'], 0.1, key="c2y", help="Y-coordinate of the center of the test contour.")
    st.session_state.params['radius2'] = st.slider("Radius (C2)", 0.0, 8.0, st.session_state.params['radius2'], 0.1, key="r2", help="Radius of the test contour (if no noise).")
    st.session_state.params['noise2'] = st.slider("Surface Noise (C2)", 0.0, 1.0, st.session_state.params['noise2'], 0.05, key="n2", help="Magnitude of random noise applied to the test contour surface.")

st.sidebar.header("Analysis Parameters")
st.session_state.params['distance_threshold'] = st.slider("Distance Threshold (mm)", 0.1, 5.0, st.session_state.params['distance_threshold'], 0.1, key="dt", help="Threshold for Surface DICE and Added Path Length (APL) calculations.")
st.session_state.params['percentile'] = st.slider("Percentile for HD (e.g., 95)", 50.0, 99.9, st.session_state.params['percentile'], 0.1, key="perc", help="Percentile used for calculating the robust Hausdorff Distance (e.g., HD95).")
st.session_state.params['num_points'] = st.slider("Sample Points per Contour", 10, 500, st.session_state.params['num_points'], 10, key="npts", help="Number of points to sample along each contour surface for surface-based metrics and noisy contour generation.")

if st.sidebar.button("Reset to Default", key="reset_button", help="Reset all parameters to their initial default values."):
    st.session_state.params = {
        'circle1_x': 0.0, 'circle1_y': 0.0, 'radius1': 4.0, 'noise1': 0.0,
        'circle2_x': 1.0, 'circle2_y': 1.0, 'radius2': 4.2, 'noise2': 0.0,
        'distance_threshold': 1.0, 'percentile': 95.0, 'num_points': 200
    }
    st.session_state.noise_cache = defaultdict(dict) 
    st.rerun()

analyzer = ContourAnalysisApp()
try:
    current_metrics = analyzer.compute_metrics(st.session_state.params, st.session_state.noise_cache)
except Exception as e:
    st.error(f"Error computing metrics: {e}. Please check input parameters.")
    # import traceback
    # st.error(f"Traceback: {traceback.format_exc()}") # Uncomment for debugging
    st.stop() 

st.header("Contour Analysis Visualizations")
if (st.session_state.params['radius1'] <=0 and st.session_state.params['noise1'] <=0 and st.session_state.params['num_points'] < 2) and \
   (st.session_state.params['radius2'] <=0 and st.session_state.params['noise2'] <=0 and st.session_state.params['num_points'] < 2) :
    st.warning("At least one contour must have a positive radius or noise with sufficient sample points to generate plots.")
else:
    try:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12)) 
        # fig.suptitle('Radiation Oncology Contour Overlap Analysis', fontsize=16, fontweight='bold') # Title is on page

        analyzer.plot_threshold_visualization(axes[0, 0], st.session_state.params)
        analyzer.plot_distance_distribution(axes[0, 1])
        analyzer.plot_surface_acceptability(axes[0, 2], st.session_state.params)
        analyzer.plot_overlap_regions(axes[1, 0], st.session_state.params)
        analyzer.plot_distance_heatmap(axes[1, 1], fig) 
        analyzer.plot_added_path_length(axes[1, 2])

        fig.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect to prevent subplot title overlap
        st.pyplot(fig)

        plot_buffer = io.BytesIO()
        fig.savefig(plot_buffer, format="png", dpi=300, bbox_inches='tight')
        plot_buffer.seek(0)
        st.sidebar.download_button(
            label="💾 Save Plot as PNG",
            data=plot_buffer,
            file_name=f"contour_analysis_{int(time.time())}.png",
            mime="image/png",
            key="save_plot_btn",
            help="Download the current set of plots as a PNG image."
        )
    except Exception as e:
        st.error(f"An error occurred during plotting: {e}")
        # import traceback
        # st.error(f"Traceback: {traceback.format_exc()}") # Uncomment for debugging


st.header("Computed Metrics")
metrics_str = analyzer.get_metrics_text()
st.text_area("Metrics Summary", metrics_str, height=480, key="metrics_summary_area", help="A summary of all computed overlap and distance metrics.")

st.sidebar.download_button(
    label="📄 Export Metrics as TXT",
    data=metrics_str,
    file_name=f"metrics_{int(time.time())}.txt",
    mime="text/plain",
    key="export_metrics_btn",
    help="Download all computed metrics as a text file."
)

st.sidebar.markdown("---")
with st.sidebar.expander("Metric Definitions & Info", expanded=False):
    st.markdown("""
    - **DICE Coefficient (Geometric)**: Measures volumetric overlap between perfect circles using their geometric formulas. (0-1, higher is better).
    - **Jaccard Index (Geometric)**: Alternative volumetric overlap metric for perfect circles.
    - **Volume Ratio (Geometric)**: Ratio of smaller to larger geometric circle area.
    - **Surface DICE**: Agreement of sampled surface points based on distance threshold.
    - **Mean Surface Distance (MSD)**: Average distance between the sampled surfaces.
    - **Hausdorff Distance (HD95)**: 95th percentile of distances between sampled surface points; robust to outliers.
    - **Max Hausdorff Distance**: Maximum distance between any point on one sampled surface to the closest point on the other.
    - **Added Path Length (APL)**: Approximate length of the test contour (sampled) that is outside the tolerance band of the reference contour.
    - **Geometric Areas**: Areas calculated from the input radii, not the sampled/noisy points.
    """)
    st.markdown("---")
    st.markdown("This tool is for educational and illustrative purposes regarding contour comparison metrics.")

st.sidebar.markdown("---")
st.sidebar.markdown("### About & License")
st.sidebar.markdown(f"""
**Developer:** Ciaran Malone
[![LinkedIn](https://content.linkedin.com/content/dam/me/business/en-us/amp/brand-site/v2/bg/LI-Bug.svg.original.svg)](https://www.linkedin.com/in/ciaranmalone/) [Connect on LinkedIn](https://www.linkedin.com/in/ciaranmalone/)

**Version:** 1.1.0

**License:** [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)
(Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International)

This tool is provided for research and educational purposes. Commercial use or redistribution for commercial purposes is prohibited. If you adapt or build upon this work, you must share your contributions under the same license.
""")