import numpy as np
from pyinstrument import Profiler

import plotly.graph_objects as go
import plotly.express as px

from gcs_lib.FreespacePolytopes import FreespacePolytopes
from gcs_lib.GraphOfConvexSets import GraphOfConvexSets
from gcs_lib.plotting import plot_obs, plot_zones

def make_rect(dims, pos, theta):
    pos = np.atleast_2d(np.array(pos).flatten()).T
    theta = np.deg2rad(theta)

    square = 0.5 * np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]]).T
    rect = np.diag(dims) @ square
    rotmat = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    return rotmat @ rect + pos

profiler = Profiler()
profiler.start()

obs_cube1 = make_rect([np.sqrt(2), np.sqrt(2)], [0, 0], 45)
obs_cube2 = make_rect([1, 1], [3.5, 4.5], 0)
r_wall = make_rect([0.5, 14], [7.25, 0], 0)
t_wall = make_rect([14, 0.5], [0, 7.25], 0)

l_wall = np.diag([-1, 1]) @ r_wall
b_wall = np.diag([1, -1]) @ t_wall

obs = [obs_cube1, obs_cube2, r_wall, t_wall, l_wall, b_wall]

# Generate random obstacles
dim_range = 3
pos_range = 6
n_obs = 10
np.random.seed(10)
for i in range(n_obs):
    obs_dims = 2*dim_range * (np.random.rand(2) - 0.5)
    obs_posns = 2*pos_range * (np.random.rand(2) - 0.5)
    theta = (np.random.rand(1)[0] - 0.5) * 360
    obs.append(make_rect(obs_dims, obs_posns, theta))

polys = FreespacePolytopes(obs, n_regions=10, grid_dims=100)

fig = go.Figure()
plot_obs(obs, fig=fig)

colors = px.colors.qualitative.Plotly[0:len(polys)]
plot_zones(polys, colors, fig=fig)

fig.update_layout(yaxis_scaleanchor="x", height=800, width=800)

try:
    graph_of_convex_sets = GraphOfConvexSets(polys)
    # x_opt, vertices_opt = graph_of_convex_sets.solve(np.array([0, -5]), np.array([0, 5]))
    x_opt, vertices_opt = graph_of_convex_sets.solve(np.array([-3.5, 5]), np.array([5.5, -3]))
    fig.add_trace(go.Scatter(
        x=x_opt[0, :],
        y=x_opt[1, :],
        line=dict(color="darkred"),
        name="Shortest Path",
        text=[f"Zone: {v}" for v in vertices_opt]
    ))
except Exception as e:
    import traceback
    traceback.print_exception(e)

fig.show()

profiler.stop()
profiler.print()