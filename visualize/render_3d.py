import itertools
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import plotly.graph_objects as go

if __package__ in (None, ""):
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.allocator import allocate_coordinates


CELLS_PER_AXIS = 3


def _cell_key(cell):
	return ",".join(str(value) for value in cell)


def _allocate_components(graph_data, seed):
	allocated = allocate_coordinates(deepcopy(graph_data))
	nodes = {
		node["id"]: node
		for node in allocated["nodes"]
		if node.get("node_type") != "container"
	}
	containers = [
		node for node in allocated["nodes"]
		if node.get("container_type") == "cluster"
	]
	rng = random.Random(seed)
	all_cells = list(itertools.product(range(CELLS_PER_AXIS), repeat=3))
	rng.shuffle(all_cells)
	groups_by_mode = {
		"simple_cubic": [],
		"body_centered_cubic": [],
	}
	for container in containers:
		mode = container["lattice_mode"]
		groups_by_mode[mode].append(container)
	for groups in groups_by_mode.values():
		rng.shuffle(groups)

	group_count = sum(len(groups) for groups in groups_by_mode.values())
	cell_count = min(group_count, len(all_cells))
	if not cell_count:
		return nodes, {}, allocated["edges"]

	target_counts = {
		mode: min(len(groups), math.floor(cell_count * len(groups) / group_count))
		for mode, groups in groups_by_mode.items()
	}
	remaining_cells = cell_count - sum(target_counts.values())
	while remaining_cells:
		candidates = [
			mode
			for mode, groups in groups_by_mode.items()
			if target_counts[mode] < len(groups)
		]
		if not candidates:
			break
		mode = max(
			candidates,
			key=lambda value: (
				cell_count * len(groups_by_mode[value]) / group_count
				- target_counts[value],
				value,
			),
		)
		target_counts[mode] += 1
		remaining_cells -= 1

	occupied_cells = all_cells[:cell_count]
	cell_groups = {cell: [] for cell in occupied_cells}
	cell_index = 0
	for mode, target in target_counts.items():
		if not target:
			continue
		mode_cells = occupied_cells[cell_index:cell_index + target]
		cell_index += target
		for index, container in enumerate(groups_by_mode[mode]):
			cell_groups[mode_cells[index % target]].append(container)

	return nodes, cell_groups, allocated["edges"]


def _component_positions(members, cell, group_index, group_count):
	cell_center = [coordinate + 0.5 for coordinate in cell]
	coordinates = {
		node["id"]: [float(value) for value in node["lattice_coordinate"]]
		for node in members
	}
	lower = [
		min(coordinate[axis] for coordinate in coordinates.values())
		for axis in range(3)
	]
	upper = [
		max(coordinate[axis] for coordinate in coordinates.values())
		for axis in range(3)
	]

	if group_count > 1:
		columns = math.ceil(group_count ** (1 / 3))
		rows = math.ceil(math.sqrt(group_count / columns))
		layers = math.ceil(group_count / (columns * rows))
		slot = (
			group_index % columns,
			(group_index // columns) % rows,
			group_index // (columns * rows),
		)
		offset = [
			((slot[axis] + 0.5) / dimension - 0.5) * 0.58
			for axis, dimension in enumerate((columns, rows, layers))
		]
		extents = [
			min(0.26, 0.58 / dimension)
			for dimension in (columns, rows, layers)
		]
	else:
		offset = [0.0, 0.0, 0.0]
		extents = [0.28, 0.28, 0.28]

	positions = {}
	for node in members:
		coordinate = coordinates[node["id"]]
		positions[node["id"]] = [
			cell_center[axis]
			+ offset[axis]
			+ (
				(coordinate[axis] - (lower[axis] + upper[axis]) / 2)
				/ ((upper[axis] - lower[axis]) / 2)
				* extents[axis]
				if upper[axis] - lower[axis] > 1e-12
				else 0.0
			)
			for axis in range(3)
		]
	return positions


def _mixed_lattice_geometry(graph_data, seed=None):
	nodes_by_id, cell_groups, edges = _allocate_components(graph_data, seed)
	sc_x = []
	sc_y = []
	sc_z = []
	bcc_x = []
	bcc_y = []
	bcc_z = []

	def add_segment(x_values, y_values, z_values, start, end):
		x_values.extend([start[0], end[0], None])
		y_values.extend([start[1], end[1], None])
		z_values.extend([start[2], end[2], None])

	for vertex in itertools.product(range(CELLS_PER_AXIS + 1), repeat=3):
		for axis in range(3):
			neighbor = list(vertex)
			neighbor[axis] += 1
			if neighbor[axis] <= CELLS_PER_AXIS:
				add_segment(sc_x, sc_y, sc_z, vertex, neighbor)

	bcc_cells = {
		cell
		for cell, groups in cell_groups.items()
		if groups and groups[0]["lattice_mode"] == "body_centered_cubic"
	}
	for cell in bcc_cells:
		center = tuple(coordinate + 0.5 for coordinate in cell)
		for corner in itertools.product((0, 1), repeat=3):
			vertex = tuple(cell[axis] + corner[axis] for axis in range(3))
			add_segment(bcc_x, bcc_y, bcc_z, center, vertex)

	sc_cells = set(cell_groups) - bcc_cells
	sc_centers = [
		tuple(coordinate + 0.5 for coordinate in cell)
		for cell in sorted(sc_cells)
	]
	bcc_centers = [
		tuple(coordinate + 0.5 for coordinate in cell)
		for cell in sorted(bcc_cells)
	]
	sc_hover = [
		f"Simple cubic cell {cell}<br>Click to expand"
		for cell in sorted(sc_cells)
	]
	bcc_hover = [
		f"BCC cell {cell}<br>Click to expand"
		for cell in sorted(bcc_cells)
	]
	lattice_vertices = list(
		itertools.product(range(CELLS_PER_AXIS + 1), repeat=3)
	)

	geometry = [
		go.Scatter3d(
			x=sc_x,
			y=sc_y,
			z=sc_z,
			mode="lines",
			line={"color": "rgba(91, 107, 121, 0.88)", "width": 5},
			hoverinfo="skip",
			name="Simple cubic bonds",
		),
		go.Scatter3d(
			x=bcc_x,
			y=bcc_y,
			z=bcc_z,
			mode="lines",
			line={"color": "rgba(221, 130, 48, 0.95)", "width": 5},
			hoverinfo="skip",
			name="BCC body-center bonds",
		),
		go.Scatter3d(
			x=[point[0] for point in lattice_vertices],
			y=[point[1] for point in lattice_vertices],
			z=[point[2] for point in lattice_vertices],
			mode="markers",
			marker={
				"size": 8,
				"color": "rgba(65, 78, 91, 0.95)",
				"line": {"color": "white", "width": 1},
			},
			hovertemplate="Shared lattice vertex<extra></extra>",
			name="Shared lattice vertices",
		),
		go.Scatter3d(
			x=[point[0] for point in sc_centers],
			y=[point[1] for point in sc_centers],
			z=[point[2] for point in sc_centers],
			mode="markers",
			marker={
				"size": 8,
				"symbol": "circle-open",
				"color": "rgba(91, 107, 121, 0.9)",
				"line": {"width": 2},
			},
			customdata=[_cell_key(cell) for cell in sorted(sc_cells)],
			text=sc_hover,
			hovertemplate="%{text}<extra></extra>",
			name="SC cells (click to expand)",
		),
		go.Scatter3d(
			x=[point[0] for point in bcc_centers],
			y=[point[1] for point in bcc_centers],
			z=[point[2] for point in bcc_centers],
			mode="markers",
			marker={
				"size": 11,
				"symbol": "diamond",
				"color": "rgba(239, 157, 69, 1)",
				"line": {"color": "white", "width": 1},
			},
			customdata=[_cell_key(cell) for cell in sorted(bcc_cells)],
			text=bcc_hover,
			hovertemplate="%{text}<extra></extra>",
			name="BCC centers (click to expand)",
		),
	]

	graph_traces_by_cell = {}
	for cell, groups in cell_groups.items():
		cell_id = _cell_key(cell)
		graph_traces_by_cell[cell_id] = []
		cell_positions = {}
		for group_index, container in enumerate(groups):
			members = [
				nodes_by_id[node_id]
				for node_id in container["contains"]
			]
			cell_positions.update(
				_component_positions(
					members,
					cell,
					group_index,
					len(groups),
				)
			)

		cell_edges = [
			edge
			for edge in edges
			if edge["source"] in cell_positions and edge["target"] in cell_positions
		]
		if cell_edges:
			edge_x = []
			edge_y = []
			edge_z = []
			for edge in cell_edges:
				source = cell_positions[edge["source"]]
				target = cell_positions[edge["target"]]
				add_segment(edge_x, edge_y, edge_z, source, target)
			graph_traces_by_cell[cell_id].append(
				go.Scatter3d(
					x=edge_x,
					y=edge_y,
					z=edge_z,
					mode="lines",
					line={"color": "rgba(47, 55, 64, 0.75)", "width": 3},
					hoverinfo="skip",
					showlegend=False,
					visible=False,
					meta=cell_id,
					name=f"Topology edges {cell_id}",
				)
			)

		cell_nodes = [
			nodes_by_id[node_id]
			for node_id in cell_positions
		]
		graph_traces_by_cell[cell_id].append(
			go.Scatter3d(
				x=[cell_positions[node["id"]][0] for node in cell_nodes],
				y=[cell_positions[node["id"]][1] for node in cell_nodes],
				z=[cell_positions[node["id"]][2] for node in cell_nodes],
				mode="markers+text",
				marker={
					"size": 6,
					"color": [
						"#4477aa"
						if str(node.get("category", "")).casefold() == "entity"
						else "#cc6677"
						for node in cell_nodes
					],
					"line": {"color": "white", "width": 0.8},
				},
				text=[node.get("label", "") for node in cell_nodes],
				textposition="top center",
				customdata=[
					[
						node["id"],
						node.get("category", ""),
						node.get("lattice_role", ""),
						node.get("lattice_index", ""),
					]
					for node in cell_nodes
				],
				hovertemplate=(
					"<b>%{text}</b><br>Category: %{customdata[1]}"
					+ "<br>ID: %{customdata[0]}"
					+ "<br>Lattice role: %{customdata[2]}"
					+ "<br>Index label: %{customdata[3]}<extra></extra>"
				),
				showlegend=False,
				visible=False,
				meta=cell_id,
				name=f"Topology particles {cell_id}",
			)
		)

	return {
		"traces": geometry,
		"graph_traces_by_cell": graph_traces_by_cell,
		"cell_groups": cell_groups,
		"bcc_cells": bcc_cells,
		"simple_cubic_bonds": geometry[0],
		"bcc_bonds": geometry[1],
		"vertices": geometry[2],
	}


def render_lattice(graph_data, output_path, seed=None):
	geometry = _mixed_lattice_geometry(graph_data, seed)
	figure = go.Figure(data=geometry["traces"])
	cell_trace_indices = {}
	for cell_id, traces in geometry["graph_traces_by_cell"].items():
		cell_trace_indices[cell_id] = []
		for trace in traces:
			cell_trace_indices[cell_id].append(len(figure.data))
			figure.add_trace(trace)

	post_script = """
var plot = document.getElementById('{plot_id}');
var cellTraces = __CELL_TRACE_INDICES__;
var expanded = new Set();
var status = document.createElement('div');
status.textContent = 'Click an SC or BCC cell marker to expand its graph particles. Click again to collapse.';
status.style.cssText = 'font: 14px sans-serif; padding: 8px 12px; color: #334155;';
plot.parentNode.insertBefore(status, plot);
plot.on('plotly_click', function(eventData) {
    var point = eventData.points && eventData.points[0];
    if (!point) return;
    var isCellMarker = point.data.name === 'SC cells (click to expand)' ||
                       point.data.name === 'BCC centers (click to expand)';
    var isExpandedParticle = point.data.name.indexOf('Topology particles ') === 0;
    if (!isCellMarker && !isExpandedParticle) return;
    var cellId = isCellMarker ? point.customdata : point.data.meta;
    var traceIndices = cellTraces[cellId] || [];
    if (traceIndices.length === 0) return;
    var shouldExpand = !expanded.has(cellId);
    Plotly.restyle(plot, {visible: shouldExpand}, traceIndices);
    if (shouldExpand) expanded.add(cellId);
    else expanded.delete(cellId);
    status.textContent = (shouldExpand ? 'Expanded ' : 'Collapsed ') +
        'cell [' + cellId + ']. Click its lattice marker or a graph particle to toggle.';
});
""".replace("__CELL_TRACE_INDICES__", json.dumps(cell_trace_indices))

	figure.update_layout(
		title=(
			"Mixed SC / BCC crystal lattice with expandable graph "
			f"({CELLS_PER_AXIS} x {CELLS_PER_AXIS} x {CELLS_PER_AXIS})"
		),
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Crystal lattice"},
		hovermode="closest",
		scene={
			"xaxis": {"title": "X", "range": [-0.25, CELLS_PER_AXIS + 0.25]},
			"yaxis": {"title": "Y", "range": [-0.25, CELLS_PER_AXIS + 0.25]},
			"zaxis": {"title": "Z", "range": [-0.25, CELLS_PER_AXIS + 0.25]},
			"aspectmode": "data",
			"dragmode": "orbit",
			"camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
		},
	)
	figure.write_html(
		output_path,
		include_plotlyjs=True,
		full_html=True,
		post_script=post_script,
	)


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph.json"
	output_path = project_dir / "docs" / "index.html"
	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	render_lattice(graph_data, output_path)
	print(f"Saved expandable mixed crystal lattice visualization to {output_path}")


if __name__ == "__main__":
	main()
