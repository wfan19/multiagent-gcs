import copy

import numpy as np
import cvxpy as cp
from scipy.spatial import HalfspaceIntersection

from .FreespacePolytopes import Polytope, FreespacePolytopes

class GraphOfConvexSets:

    polys: FreespacePolytopes
    mat_edges: np.ndarray
    mat_graph_adj: np.ndarray

    def __init__(self, polys: FreespacePolytopes):
        self.polys = polys
        self.mat_edges, self.mat_graph_adj = self._get_mat_edges(polys)
            
    @staticmethod
    def _get_mat_edges(polys: FreespacePolytopes):
        # Build adjacency matrix
        n_polys = len(polys) #TODO: It's ugly to just get the length of one of the properties
        mat_graph_adj = np.zeros((n_polys, n_polys))
        for i, poly_i in enumerate(polys):
            # For each polygon, loop through all polygons to find others that overlap
            for j, poly_j in enumerate(polys):
                # Pass if on diagonal: nodes are not connected to self
                if j == i:
                    continue
                    
                # Check if any vertices of this polygon_j overlaps with polygon_i
                # Note we use 1e-5 here because sometimes an overlapping polytope's vertex is on the edge of the polytope it overlaps.
                verts_overlapping = np.max(poly_j.A @ poly_i.vertices - poly_j.b, axis=0) <= 1e-5
                overlapping = np.any(verts_overlapping) # If any vertex overlaps with the current polytope, then there is overlap.
                if overlapping:
                    # Set connection to 1 in adjacency matrix
                    mat_graph_adj[i, j] = 1
                    mat_graph_adj[j, i] = 1
                    
        # Build the list of edges from the adjacency matrix
        mat_edges = []
        for i, row in enumerate(mat_graph_adj):
            for j, target_edge in enumerate(row):
                if target_edge > 0:
                    mat_edges.append(np.array([i, j]))
        mat_edges = np.array(mat_edges)
        return mat_edges, mat_graph_adj

    ## Helper functions for building the optimization problem
    # Get the set of incoming and outgoing edges for a given vertex v
    @staticmethod
    def _get_io_edges(v, edges):
        """Get indices of edges that are inbound or outbound from a given vertex

        Args:
            v (int): Index of vertex of interest
            edges (np.ndarray): Edge matrix

        Raises:
            ValueError: Neither incoming nor outgoing edges are found for the specified node

        Returns:
            Tuple[np.ndarray, np.ndarray]: Returns two lists, respectively containing indices for inbound and outbound edges
        """
        edges_in = np.where(edges[:, 1] == v)[0] # Edges that end with our vertex are coming in
        edges_out = np.where(edges[:, 0] == v)[0] # Edges that begin with our vertex are going out
        
        if (not np.any(edges_in)) and (not np.any(edges_out)):
            raise ValueError(f"Neither incoming nor outgoing edges found for node {v} in the following edges:\n{edges}")
        
        return edges_in, edges_out

    def _solve_gcs_perspective(self, x_0, x_goal, polys_st, mat_edges_st, s, t):
        """Build and solve the shortest-path-on-GCS problem, but reformulated through perspective transforms.
        - Contains the actual GCS mathematical problem setup and solution.

        Args:
            x_0 (np.ndarray): 1d array of start point position
            x_goal (np.ndarray): 1d array of end point position
            polys_st (FreespacePolytopes): Polytopes including start and endpoint
            mat_edges_st (np.ndarray): Edge matrix of graph including start and end edges

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Returns decision variables z, z_prime, and y
        """

        # Define our epsilon based on machine precision epsilon. 100 comes from the drake GCS impl.
        eps = 100 * np.finfo(float).eps
        n_decimals = np.round(np.abs(np.log10(eps)))

        # Define decision variables
        num_edges = len(mat_edges_st)
        z = cp.Variable((self.polys.n_dims, num_edges), name="z")
        z_prime = cp.Variable((self.polys.n_dims, num_edges), name="zprime")
        # Unfortunately cp.perspective cannot take elements of a vector for its 's' argument
        # So we have to manually make a list of individual y variables that we feed to cp.perspective one by one.
        ys = []
        for i in range(num_edges):
            y = cp.Variable(1, boolean=True, name=f"y{i}")
            # y = cp.Variable(1, nonneg=True, name=f"y{i}")
            ys.append(y)

        cost = 0
        constr = []

        ## Construct costs and constraints that are per-edge
        for e, edge in enumerate(mat_edges_st):
            u, v = edge
            # Sum of squared distace along each edge connecting the vertices in each polytope
            f = cp.norm(z[:, e] - z_prime[:, e])
            f_recession = 0 * f
            cost += cp.perspective(f, ys[e], f_recession) # Eqn 5.5a
            
            # Edge constraint: edge must belong to the perspective cone of the free polytopes
            constr += [polys_st[u].A @ z[:, e] - cp.vec(ys[e] * polys_st[u].b) <= eps]
            constr += [polys_st[u].A @ z_prime[:, e] - cp.vec(ys[e] * polys_st[u].b) <= eps] # Eqn 5.5d
            
        ## Construct costs and constraints that are per-vertex
        # for i, (A_v, b_v) in enumerate(zip(As_st, bs_st)):
        for i, poly in enumerate(polys_st):
            if (i == s) or (i == t):
                continue
            # Flow constraints
            edges_in, edges_out = self._get_io_edges(i, mat_edges_st)
            
            # Constraint 5.5c, the part for y_e, and the mutual-equivalence part of 5.5b
            y_sum_in, y_sum_out = 0, 0 # Manually sum y by looping :( sadly because ys is in silly(python) vector form
            for y in [ys[i] for i in edges_in]:
                y_sum_in += y
            for y in [ys[i] for i in edges_out]:
                y_sum_out += y
            constr += [y_sum_in == y_sum_out]
            
            constr += [cp.sum(z_prime[:, edges_in], axis=1) == cp.sum(z[:, edges_out], axis=1)] # the z-part of 5.5c

            ## Eliminate 2-cycles
            # - GCS Control paper (Motion Planning around Obstacles with Convex Optimization) Appendix A.1
            # - Constrain all nodes to only have one output
            # - Example implementation: https://github.com/RobotLocomotion/drake/blob/386ef0b4985ee636777324f4dab94e0141aca832/geometry/optimization/graph_of_convex_sets.cc#L798
            # This constraint should be made redundant by the 2-cycle checks below but instead it appears neccessary.
            # This may be a sign that the 2-cycle constraints below are now working properly.
            constr += [y_sum_out <= 1]
            
            # Now forbid all 2-cycles in/out of this vertex. 
            # - See Appendix A.1, or Line 819 in the Drake implementation.
            # TODO: Is there a redundancy issue here? Do we over-add constraints, both (ij) and (ji)?
            # NOTE: Introducing this constraint seems overconstrain the system.
            # - Causes single agent solves to fail, or often choose visibly suboptimal paths
            # - Probably part of why multiagent solves fail as well.

            for e_out in edges_out:
                for e_in in edges_in:
                    if mat_edges_st[e_in, 0] == mat_edges_st[e_out, 1]:
                        # breakpoint()
                        # Cyclical edge pair detected. Add flow constraint to prevent a cyclical path.
                        flow_diff = y_sum_out - ys[e_in] - ys[e_out] 
                        constr += [flow_diff >= -eps]
                        
                        # Spatial flow constraint:
                        # - Mentioned in passing in Appendix A.1, or Line 824 in Drake implementation.
                        # - Not neccessary but gives additional tightness for the convex relaxation - makes solving faster.
                        # - Keep in mind {z, z_prime} in GCS paper is {y, z} in Drake impl.
                        v_spatial_flow = cp.sum(z_prime[:, edges_in], axis=1) - z_prime[:, e_in] - z[:, e_out]
                        constr += [poly.A @ v_spatial_flow - cp.vec(flow_diff * poly.b) <= eps]
                        
                        
        ## Start/end point constraints
        _, edges_out_of_s = self._get_io_edges(s, mat_edges_st)
        edges_into_t, _ = self._get_io_edges(t, mat_edges_st)

        # Constraint 5.5b
        y_sum_s, y_sum_t = 0, 0
        for y in [ys[i] for i in edges_out_of_s]:
            y_sum_s += y
        for y in [ys[i] for i in edges_into_t]:
            y_sum_t += y
        constr += [y_sum_s == y_sum_t, y_sum_s == 1]

        # Constrain the start and end points to be at their defined positions
        # Boundary condition 1f in the GCS Control Paper
        constr += [cp.sum(z[:, edges_out_of_s], axis=1) == x_0, cp.sum(z_prime[:, edges_into_t], axis=1) == x_goal]

        # y_sum_total = 0
        # for y in ys:
        #     y_sum_total += y
        # cost += y_sum_total

        ## Solve the problem!!
        prob = cp.Problem(cp.Minimize(cost), constr)
        mosek_params = {"MSK_IPAR_INFEAS_REPORT_AUTO": "MSK_ON"}
        prob.solve(solver="MOSEK", verbose=True, mosek_params=mosek_params)
        if prob.status is not None and not prob.status == "optimal":
            raise RuntimeError("GCS failed to find a solution")

        ys_val = np.round(np.array([y.value for y in ys])).flatten().astype(np.bool_)

        return z.value, z_prime.value, ys_val


    def solve(self, x_0: np.ndarray, x_goal: np.ndarray):
        """Public-facing wrapper for intializing and postprocessing the GCS shortest-path problem.
        1. Given start and end points, identify parent polytopes and add vertices/edges
        2. Solve the problem
        3. Order the returned vertices into a properly continuous path
        4. Reconstruct positions x from the optimization variables z and z_prime

        Args:
            x_0 (np.ndarray): Start point
            x_goal (np.ndarray): End point

        Returns:
            np.ndarray: The sequence of points x which form the optimal path
        """
        # Find polytopes (graph nodes) which contain the start and end points
        s_poly, t_poly = -1, -1
        for i, poly in enumerate(self.polys):
            if np.all(poly.A @ np.atleast_2d(x_0).T - poly.b <= 0) and (s_poly == -1):
                s_poly = i
            if np.all(poly.A @ np.atleast_2d(x_goal).T - poly.b <= 0) and (t_poly == -1):
                t_poly = i

        # Insert additional start and endpoint nodes into the graph
        # Note: If we directly impose constraints on position of points in the regions, this makes problems overdefined and unsolvable.
        # Start and end point constraints call for the creation of new graph nodes to avoid overconstraining the problem.
        # Currently, we just insert a node that's a copy of the node with the polytope the start/end points are in.
        n_polys = len(self.polys)
        s, t = n_polys, n_polys + 1
        start_edge, end_edge = [s, s_poly], [t_poly, t]
        # TODO: If we have a better underlying graph structure, then these can be formulated as just adding two nodes and edges. This is kinda hacky
        mat_edges_st = np.vstack([self.mat_edges, start_edge, end_edge])
        polys_st = copy.deepcopy(self.polys)
        polys_st.extend([polys_st[s_poly], polys_st[t_poly]])

        ## Build and solve the mathematical problem!
        np.set_printoptions(suppress=True)
        z, z_prime, y = self._solve_gcs_perspective(x_0, x_goal, polys_st, mat_edges_st, s, t)
        print(f"Z:\n{z.T}")
        print(f"Z prime:\n{z_prime.T}")
        print(f"y:\n{np.atleast_2d(y).T}")

        ## Reconstruct the point positions x from the flow variables z, z_prime, and y
        xs = np.zeros((len(x_0), len(polys_st)))
        for i in range(len(polys_st) - 2):
            _, edges_out = self._get_io_edges(i, mat_edges_st)
            
            z_sum = np.sum(z[:, edges_out], axis=1)
            y_sum = np.sum(y[edges_out])
            
            # Eqn. 5.7 
            x_v = z_sum / y_sum if y_sum > 1e-5 else np.zeros(z_sum.shape) * np.nan
            xs[:, i] = x_v
            
        # Independently reconstruct start and end point positions (Eqn. 5.6)
        _, edges_out_of_s = self._get_io_edges(s, mat_edges_st)
        edges_into_t, _ = self._get_io_edges(t, mat_edges_st)
        xs[:, -2] = np.sum(z[:, edges_out_of_s], axis=1)
        xs[:, -1] = np.sum(z_prime[:, edges_into_t], axis=1)

        print(f"xs:\n{xs.T}")
        
        ## Reorder the vertices into a sequence following the order of the path
        # For example, e_edges_in_path may be [3, 1, 5]
        # But if edge_3 = [3, 2], edge_1 = [2, 4], and edge_5 = [1, 3]
        # Then the actual path should be [1, 3], [3, 2], [2, 4]
        # So e_edges_ordered = [5, 3, 1].
        e_edges_in_path = np.where(y)[0] # Incorrectly ordered list of edge id's (e) for the edges actually in the path
        edges_in_path = mat_edges_st[y, :] # List of the edges themeselves (vertex pairs)

        i_edges_ordered = np.zeros(e_edges_in_path.shape, dtype=np.int_) # List of indices for the correct ordering of the edge ids

        next_vertex = s
        for i in range(len(e_edges_in_path)):
            # Index from reduced path edge list which has the edge
            # NOTE: This is currently O(n^2) where n = path length. (for loop * np.where) - is there a better way?
            i_current_edge = np.where(edges_in_path[:, 0] == next_vertex)[0] # Find index of edge which begins with the current vertex
            current_edge = edges_in_path[i_current_edge, :]

            i_edges_ordered[i] = i_current_edge
            next_vertex = current_edge[:, 1] # The end vertex of this edge is the start vertex of the next one
            
        e_edges_ordered = e_edges_in_path[i_edges_ordered]
        edges_ordered = mat_edges_st[e_edges_ordered, :]
        all_vertices_but_last, last_vertex = edges_ordered[:, 0].flatten(), [edges_ordered[-1, 1]]
        v_ordered = np.concatenate([all_vertices_but_last, last_vertex])

        # Select vertex point positions in the correct order, save and return
        x_out = xs[:, v_ordered]
        return x_out, v_ordered