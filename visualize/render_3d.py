import itertools
import random
from pathlib import Path

import plotly.graph_objects as go


CELLS_PER_AXIS = 2


def _mixed_lattice_geometry(seed=None):
	rng = random.Random(seed)
	cells = list(itertools.product(range(CELLS_PER_AXIS), repeat=3))
	rng.shuffle(cells)
	bcc_cells = set(cells[: len(cells) // 2])

	sc_x = []
	sc_y = []
	sc_z = []
	bcc_x = []
	bcc_y = []
	bcc_z = []
	center_x = []
	center_y = []
	center_z = []
	center_labels = []

	def add_segment(x_values, y_values, z_values, start, end):
		x_values.extend([start[0], end[0], None])
		y_values.extend([start[1], end[1], None])
		z_values.extend([start[2], end[2], None])

	for vertex in itertools.product(range(CELLS_PER_AXIS + 1), repeat=3):
		for axis in range(3):
			neighbor = list(vertex)
			neighbor[axis] += 1
			if neighbor[axis] > CELLS_PER_AXIS:
				continue
			add_segment(sc_x, sc_y, sc_z, vertex, neighbor)

	for cell in sorted(bcc_cells):
		center = tuple(coordinate + 0.5 for coordinate in cell)
		center_x.append(center[0])
		center_y.append(center[1])
		center_z.append(center[2])
		center_labels.append("BCC center: " + ", ".join(map(str, cell)))
		for corner in itertools.product((0, 1), repeat=3):
			vertex = tuple(cell[axis] + corner[axis] for axis in range(3))
			add_segment(bcc_x, bcc_y, bcc_z, center, vertex)

	return {
		"bcc_cells": bcc_cells,
		"simple_cubic_bonds": go.Scatter3d(
			x=sc_x,
			y=sc_y,
			z=sc_z,
			mode="lines",
			line={"color": "rgba(91, 107, 121, 0.88)", "width": 5},
			hoverinfo="skip",
			name="Simple cubic bonds",
		),
		"bcc_bonds": go.Scatter3d(
			x=bcc_x,
			y=bcc_y,
			z=bcc_z,
			mode="lines",
			line={"color": "rgba(221, 130, 48, 0.95)", "width": 5},
			hoverinfo="skip",
			name="BCC body-center bonds",
		),
		"vertices": go.Scatter3d(
			x=[
				point[0]
				for point in itertools.product(
					range(CELLS_PER_AXIS + 1), repeat=3
				)
			],
			y=[
				point[1]
				for point in itertools.product(
					range(CELLS_PER_AXIS + 1), repeat=3
				)
			],
			z=[
				point[2]
				for point in itertools.product(
					range(CELLS_PER_AXIS + 1), repeat=3
				)
			],
			mode="markers",
			marker={
				"size": 8,
				"color": "rgba(65, 78, 91, 0.95)",
				"line": {"color": "white", "width": 1},
			},
			hovertemplate="Shared lattice vertex<extra></extra>",
			name="Shared lattice vertices",
		),
		"centers": go.Scatter3d(
			x=center_x,
			y=center_y,
			z=center_z,
			mode="markers",
			marker={
				"size": 11,
				"symbol": "diamond",
				"color": "rgba(239, 157, 69, 1)",
				"line": {"color": "white", "width": 1},
			},
			customdata=center_labels,
			hovertemplate="%{customdata}<extra></extra>",
			name="BCC body-center atoms",
		),
	}


def render_lattice(output_path, seed=None):
	geometry = _mixed_lattice_geometry(seed)
	figure = go.Figure(
		data=[
			geometry["simple_cubic_bonds"],
			geometry["bcc_bonds"],
			geometry["vertices"],
			geometry["centers"],
		]
	)
	figure.update_layout(
		title="Random mixed SC / BCC crystal lattice (2 x 2 x 2)",
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Crystal lattice"},
		hovermode="closest",
		scene={
			"xaxis": {"title": "X", "range": [-0.25, 2.25]},
			"yaxis": {"title": "Y", "range": [-0.25, 2.25]},
			"zaxis": {"title": "Z", "range": [-0.25, 2.25]},
			"aspectmode": "data",
			"dragmode": "orbit",
			"camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
		},
	)
	figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def main():
	project_dir = Path(__file__).resolve().parents[1]
	output_path = project_dir / "docs" / "index.html"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	render_lattice(output_path)
	print(f"Saved mixed crystal lattice visualization to {output_path}")


if __name__ == "__main__":
	main()
