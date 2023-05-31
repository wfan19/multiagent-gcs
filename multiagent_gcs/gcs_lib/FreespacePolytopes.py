from typing import List
from dataclasses import dataclass
import warnings

import numpy as np
import cvxpy as cp

from scipy.spatial import HalfspaceIntersection
from scipy.spatial import distance

@dataclass
class Polytope():
    A: np.ndarray
    b: np.ndarray
    vertices: np.ndarray

class FreespacePolytopes(list):
    """An implementation of the IRIS free-space convex partitioning algorithm
    as described by Diets et al, 2014. 
    """

    obstacles: List[np.ndarray]

    def __init__(self, obstacles: List[np.ndarray], n_regions=10, grid_dims=10):
        self.obstacles = obstacles
        polys_list = self._convex_freespace_decomp(n_regions, grid_dims)
        
        # Pass the list of Polytopes to the Python list parent class data storage
        super(FreespacePolytopes, self).__init__(polys_list)

    def _validate_dims(self, dim: int = None):
        v_obs_dims = np.array([obstacle.shape[0] for obstacle in self.obstacles])
        assert np.all(v_obs_dims == v_obs_dims[0]), "Obstacle list has inconsistent dimensionality"

        if dim is not None:
            assert np.all(v_obs_dims == dim), f"Obstacle list does not match expected dimensionality {dim}"

    @property
    def n_dims(self):
        self._validate_dims()
        return self.obstacles[0].shape[0]

    def _find_closest_point_to_ellipse(self, C, d, vertices):
        """Given a list of vertices that define a poly, find point in poly closest to ellipse

        Args:
            C (np.ndarray): Ellipse matrix
            d (np.ndarray): Ellipse center point
            points (np.ndarray): list of points to search through

        Returns:
            np.ndarray: the closest point to the ellipse
        """
        # Transform obstacle points from world frame into world-space into ball-space
        obs_j_ball_space = np.linalg.inv(C) @ (vertices - d)
        
        # Find closest point in ball-space
        x_tilde = cp.Variable((self.n_dims, 1))
        w = cp.Variable((vertices.shape[1], 1), nonneg=True)
        obj = cp.Minimize(cp.sum_squares(x_tilde)) # Find point which is closest to origin
        constr = [obs_j_ball_space @ w == x_tilde, cp.sum(w) == 1] # Constrain x to inside of obstacle via convex combination of vertices
        prob = cp.Problem(obj, constr)
        prob.solve()

        # Transform the closest point in ball-space back to world space
        x_star_j = C @ x_tilde.value + d
        return x_star_j

    def _find_hyperplanes(self, C, d):
        # Step 1: Order all obstacles by distance of closest vertex to ellipse, shortest first.
        min_dists = np.array([np.min(np.linalg.norm(obs_i - d, axis=0)) for obs_i in self.obstacles])
        i_sort_obs_by_dist = np.argsort(min_dists).astype(np.int_)
        
        obs_remaining = [self.obstacles[i] for i in i_sort_obs_by_dist]
        A = []
        b = []
        while obs_remaining:
            obs_closest = obs_remaining[0]
            x_star_j = self._find_closest_point_to_ellipse(C, d, obs_closest)

            # Compute plane tangent to ellipse and passing through point
            # Deits et al 2014 Equation 6 and 7
            C_inv = np.linalg.inv(C)
            a_i = (2 * C_inv @ C_inv.T @ (x_star_j - d)).T
            b_i = a_i @ x_star_j

            i_obs_to_exclude = [0]
            for i, obs in enumerate(obs_remaining):
                # Check if any further obstacles are also excluded by this plane
                if np.all(a_i @ obs - b_i >= -1e-5):
                    i_obs_to_exclude.append(i)
            
            # Remove the redundant obstacles
            i_obs_to_exclude = frozenset(i_obs_to_exclude)
            obs_remaining = [obs for i, obs in enumerate(obs_remaining) if i not in i_obs_to_exclude]

            A.append(a_i.flatten())
            b.append(b_i.flatten())

        A = np.array(A)
        b = np.array(b)
        return A, b

    def _maximize_ellipsoid(self, A, b):
        """Find the largest ellipsoid that can fit within the bounding planes defined by A, b.
        An implementation of the semidefinite program in Eq. 10 in Diets 2014.

        Args:
            A (np.ndarray): Bounding line definition matrix
            b (np.ndarray): Bounding line definition vector

        Returns:
            C: Ellipse definition matrix: Cx for x in unit circle transforms the circle into an ellipse
            d: Centerpoint of ellipse
        """
        # Find the largest ellipsoid that can fit within the given set of planes
        # Implementation of the semidefinite pr"ogram in Eq. 10 in Diets 2014.
        n_dims = self.n_dims
        C = cp.Variable((n_dims, n_dims), PSD=True)
        d = cp.Variable((n_dims, 1))
        
        obj = cp.Maximize(cp.log_det(C))
        constr = []
        for i, a in enumerate(A):
            constr += [cp.norm(a @ C) + a @ d <= b[i]]
        
        prob = cp.Problem(obj, constr)
        try:
            # prob.solve(solver="MOSEK")
            prob.solve("SCS")
        except Exception as e:
            import traceback
            traceback.print_exception(e)
            breakpoint()
        
        return C.value, d.value

    def _find_poly(self, seed_pt: np.ndarray, stop_rate: float = 0.05, r_start=0.1):
        """An implementation of the IRIS free-space convex partitioning algorithm from Diets et al 2014.
        Given an initial seed point, find a large polytope in free-space by alternating between
        growing an ellipse from the seed point up bounding planes, and then redefining bounding planes
        based on the new ellipse shape.

        Args:
            startpoint (np.ndarray): Seed point in configuration space to start the algorithm. A ellipse (circle) of radius r_start is placed there
            stop_rate (float, optional): If growth of ellipse volume is less than this percent then algorithm will terminate. Defaults to 0.1.
        """
        self._validate_dims(len(seed_pt))
        d_0 = np.atleast_2d(seed_pt.flatten()).T
        C_0 = r_start * np.eye(self.n_dims)
        
        last_C = C_0
        while True:
            A, b = self._find_hyperplanes(C_0, d_0)
            C, d = self._maximize_ellipsoid(A, b)
            
            f_log_det = lambda A: np.log(np.linalg.det(A))
            if (f_log_det(C) - f_log_det(last_C)) / f_log_det(last_C) <= stop_rate:
                break
            last_C = C
        
        return A, b, d
    
    def _convex_freespace_decomp(self, n_regions, grid_dims) -> List[Polytope]:
        """Decompose the overall obstacle space into individual convex regions.

        Returns:
            As: list of A matrices
            bs: list of b vectors
        """
        ## Create freespace polytope search seed-points using heuristic described in Deits et al 2015 II.A
        # Create list of obstacle points, additionally with points at their centroids. We want to build away from these.
        obs_points = np.hstack([obs for obs in self.obstacles])
        mid_points = np.array([np.mean(obs, axis=1) for obs in self.obstacles]).T
        obs_points = np.concatenate([obs_points, mid_points], axis=1) # This works really well, but disable right now for debug purposes

        # Create a coarse grid across the space
        scale_bounds = 0.8
        min_coords = np.amin(obs_points, axis=1) * scale_bounds
        max_coords = np.amax(obs_points, axis=1) * scale_bounds
        grid_coords = np.linspace(min_coords, max_coords, grid_dims).T
        meshes = np.meshgrid(*grid_coords)
        mesh_points = np.array([mesh.flatten() for mesh in meshes])

        # Find the mesh point which maximizes distance from obstacles
        dists = distance.cdist(mesh_points.T, obs_points.T)
        i_furthest = np.argmax(np.amin(dists, axis=1))

        next_seed_posn = mesh_points[:, i_furthest]

        polys_list = []
        for i in range(n_regions):
            print(f"Seed position: {next_seed_posn}")
            # Find the polytope given the current start position!
            A_i, b_i, d_i = self._find_poly(next_seed_posn)

            if np.linalg.matrix_rank(A_i) < self.n_dims:
                warnings.warn(f"Warning: Skipping invalid seed point {next_seed_posn}")
                # We failed to find a poly, probably because the point is inside of an obstacle
                # Let's take out this current mesh point and try another one.
                mesh_points = np.delete(mesh_points, i_furthest, 1)
                dists = distance.cdist(mesh_points.T, obs_points.T)
                i_furthest = np.argmax(np.amin(dists, axis=1))
                next_seed_posn = mesh_points[:, i_furthest]
                continue

            # Find vertices of the polytope given the separating planes
            try:
                vertices_i = HalfspaceIntersection(np.hstack([A_i, -b_i]), d_i.flatten()).intersections.T
            except:
                raise RuntimeError("Bad polytope solution did not get caught earlier in the process! I don't want to handle it here.") 

            # If in 2D: reorder the halfspace intersection points by their polar angle
            # - This way we make sure we're always plotting the points going around the perimeter
            # - Does not apply to >2D polytopes - no clear "one-dimensional" ordering of vertices / doesn't matter for plotting anyways.
            # TODO: is there still a way to plot a low-dimensional representation of the free polytopes, given a 4-d partitioning of the space?
            if self.n_dims == 2:
                poly_center = np.mean(vertices_i, axis=1)
                thetas = np.arctan2(vertices_i[1, :] - poly_center[1], vertices_i[0, :] - poly_center[0])
                vertices_i = vertices_i[:, np.argsort(thetas)]

            # Remove mesh points in current poly from future searches. 
            mesh_in_poly = np.any(A_i @ mesh_points - b_i > -1e-5, axis=0)
            mesh_points = mesh_points[:, mesh_in_poly]
            
            obs_points = np.hstack([obs_points, vertices_i])
            dists = distance.cdist(mesh_points.T, obs_points.T)
            i_furthest = np.argmax(np.amin(dists, axis=1))
            next_seed_posn = mesh_points[:, i_furthest]

            polys_list.append(Polytope(A_i, b_i, vertices_i))

        return polys_list