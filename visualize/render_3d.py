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


MIN_CELLS_PER_AXIS = 3


def _site_key(kind, coordinate):
	return kind + ":" + ",".join(f"{value:g}" for value in coordinate)


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
	rng.shuffle(containers)
	cells_per_axis = max(
		MIN_CELLS_PER_AXIS,
		math.ceil(len(containers) ** (1 / 3)),
	)
	cells = list(itertools.product(range(cells_per_axis), repeat=3))
	rng.shuffle(cells)
	cell_groups = {
		cells[index]: container
		for index, container in enumerate(containers)
	}
	return nodes, cell_groups, allocated["edges"], cells_per_axis


def _component_site_positions(members, cell):
	corners = list(
		itertools.product(*(range(value, value + 2) for value in cell))
	)
	node_coordinates = {
		node["id"]: tuple(float(value) for value in node["lattice_coordinate"])
		for node in members
	}
	vertex_nodes = [
		node for node in members
		if node.get("lattice_site") != "body_center"
	]
	vertex_nodes.sort(
		key=lambda node: (node_coordinates[node["id"]], node["id"])
	)
	center_nodes = [
		node for node in members
		if node.get("lattice_site") == "body_center"
	]
	center_nodes.sort(key=lambda node: node["id"])

	positions = {}
	for index, node in enumerate(vertex_nodes):
		coordinate = corners[index % len(corners)]
		repeated = index // len(corners)
		offset = ((repeated + 1) // 2) * (0.08 if repeated % 2 == 0 else -0.08)
		positions[node["id"]] = {
			"kind": "vertex",
			"coordinate": coordinate,
			"position": [float(value) for value in coordinate],
			"label_offset": offset,
		}

	center_coordinate = tuple(value + 0.5 for value in cell)
	for index, node in enumerate(center_nodes):
		positions[node["id"]] = {
			"kind": "center",
			"coordinate": center_coordinate,
			"position": [float(value) for value in center_coordinate],
			"label_offset": (index - (len(center_nodes) - 1) / 2) * 0.08,
		}
	return positions


def _mixed_lattice_geometry(graph_data, seed=None):
	nodes_by_id, cell_groups, edges, cells_per_axis = _allocate_components(
		graph_data,
		seed,
	)
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

	for vertex in itertools.product(range(cells_per_axis + 1), repeat=3):
		for axis in range(3):
			neighbor = list(vertex)
			neighbor[axis] += 1
			if neighbor[axis] <= cells_per_axis:
				add_segment(sc_x, sc_y, sc_z, vertex, neighbor)

	bcc_cells = {
		cell
		for cell, container in cell_groups.items()
		if container["lattice_mode"] == "body_centered_cubic"
	}
	for cell in bcc_cells:
		center = tuple(coordinate + 0.5 for coordinate in cell)
		for corner in itertools.product((0, 1), repeat=3):
			vertex = tuple(cell[axis] + corner[axis] for axis in range(3))
			add_segment(bcc_x, bcc_y, bcc_z, center, vertex)

	lattice_vertices = list(
		itertools.product(range(cells_per_axis + 1), repeat=3)
	)
	bcc_centers = [
		tuple(coordinate + 0.5 for coordinate in cell)
		for cell in sorted(bcc_cells)
	]
	site_contents = {}
	node_positions = {}
	for cell, container in cell_groups.items():
		members = [
			nodes_by_id[node_id]
			for node_id in container["contains"]
		]
		positions = _component_site_positions(members, cell)
		for node in members:
			placement = positions[node["id"]]
			key = _site_key(placement["kind"], placement["coordinate"])
			site_contents.setdefault(key, []).append(node)
			node_positions[node["id"]] = placement["position"]

	vertex_contents = {
		_site_key("vertex", vertex): site_contents.get(
			_site_key("vertex", vertex), []
		)
		for vertex in lattice_vertices
	}
	center_contents = {
		_site_key("center", center): site_contents.get(
			_site_key("center", center), []
		)
		for center in bcc_centers
	}
	all_sites = {**vertex_contents, **center_contents}
	site_trace_indices = {}
	topology_traces = []
	for key, site_nodes in all_sites.items():
		if not site_nodes:
			continue
		site_trace_indices[key] = []
		site_node_ids = {node["id"] for node in site_nodes}
		site_edges = [
			edge
			for edge in edges
			if edge["source"] in site_node_ids or edge["target"] in site_node_ids
		]
		if site_edges:
			edge_x = []
			edge_y = []
			edge_z = []
			for edge in site_edges:
				add_segment(
					edge_x,
					edge_y,
					edge_z,
					node_positions[edge["source"]],
					node_positions[edge["target"]],
				)
			site_trace_indices[key].append(len(topology_traces))
			topology_traces.append(
				go.Scatter3d(
					x=edge_x,
					y=edge_y,
					z=edge_z,
					mode="lines",
					line={"color": "rgba(47, 55, 64, 0.78)", "width": 3},
					hoverinfo="skip",
					showlegend=False,
					visible=False,
					meta=key,
					name=f"Topology edges {key}",
				)
			)

		site_x = []
		site_y = []
		site_z = []
		for index, node in enumerate(site_nodes):
			position = node_positions[node["id"]]
			offset = (index - (len(site_nodes) - 1) / 2) * 0.08
			site_x.append(position[0] + offset)
			site_y.append(position[1] + offset)
			site_z.append(position[2] + offset)
		site_trace_indices[key].append(len(topology_traces))
		topology_traces.append(
			go.Scatter3d(
				x=site_x,
				y=site_y,
				z=site_z,
				mode="markers+text",
				marker={
					"size": 6,
					"color": [
						"#4477aa"
						if str(node.get("category", "")).casefold() == "entity"
						else "#cc6677"
						for node in site_nodes
					],
					"line": {"color": "white", "width": 0.8},
				},
				text=[node.get("label", "") for node in site_nodes],
				textposition="top center",
				customdata=[
					[
						node["id"],
						node.get("category", ""),
						node.get("lattice_role", ""),
						node.get("lattice_index", ""),
					]
					for node in site_nodes
				],
				hovertemplate=(
					"<b>%{text}</b><br>Category: %{customdata[1]}"
					+ "<br>ID: %{customdata[0]}"
					+ "<br>Lattice role: %{customdata[2]}"
					+ "<br>Index label: %{customdata[3]}<extra></extra>"
				),
				showlegend=False,
				visible=False,
				meta=key,
				name=f"Topology particles {key}",
			)
		)

	def site_hover(kind, coordinate):
		key = _site_key(kind, coordinate)
		count = len(all_sites.get(key, []))
		return f"{kind.title()}<br>{count} nested graph particles<br>Click to expand"

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
			customdata=[_site_key("vertex", point) for point in lattice_vertices],
			text=[site_hover("vertex", point) for point in lattice_vertices],
			hovertemplate="%{text}<extra></extra>",
			name="Shared lattice vertices",
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
			customdata=[_site_key("center", point) for point in bcc_centers],
			text=[site_hover("center", point) for point in bcc_centers],
			hovertemplate="%{text}<extra></extra>",
			name="BCC centers",
		),
	]

	return {
		"traces": geometry,
		"topology_traces": topology_traces,
		"site_trace_indices": site_trace_indices,
		"cell_groups": cell_groups,
		"bcc_cells": bcc_cells,
		"cells_per_axis": cells_per_axis,
		"simple_cubic_bonds": geometry[0],
		"bcc_bonds": geometry[1],
		"vertices": geometry[2],
		"centers": geometry[3],
		"site_contents": all_sites,
	}


def render_lattice(graph_data, output_path, seed=None):
	geometry = _mixed_lattice_geometry(graph_data, seed)
	figure = go.Figure(data=geometry["traces"])
	site_trace_indices = {}
	for key, traces in geometry["site_trace_indices"].items():
		site_trace_indices[key] = [len(figure.data) + index for index in traces]
	figure.add_traces(geometry["topology_traces"])

	post_script = """
var plot = document.getElementById('{plot_id}');
var siteTraces = __SITE_TRACE_INDICES__;
var expanded = new Set();
var status = document.createElement('div');
status.textContent = 'Click a Shared lattice vertex or BCC center to expand its nested graph particles.';
status.style.cssText = 'font: 14px sans-serif; padding: 8px 12px; color: #334155;';
plot.parentNode.insertBefore(status, plot);
plot.on('plotly_click', function(eventData) {
    var point = eventData.points && eventData.points[0];
    if (!point) return;
    var isLatticeSite = point.data.name === 'Shared lattice vertices' ||
                        point.data.name === 'BCC centers';
    var isExpandedParticle = point.data.name.indexOf('Topology particles ') === 0;
    if (!isLatticeSite && !isExpandedParticle) return;
    var siteKey = isLatticeSite ? point.customdata : point.data.meta;
    var traceIndices = siteTraces[siteKey] || [];
    if (traceIndices.length === 0) return;
    var shouldExpand = !expanded.has(siteKey);
    Plotly.restyle(plot, {visible: shouldExpand}, traceIndices);
    if (shouldExpand) expanded.add(siteKey);
    else expanded.delete(siteKey);
    status.textContent = (shouldExpand ? 'Expanded ' : 'Collapsed ') +
        siteKey + '. Click the site or one of its graph particles to toggle.';
});
""".replace("__SITE_TRACE_INDICES__", json.dumps(site_trace_indices))

	cells_per_axis = geometry["cells_per_axis"]
	figure.update_layout(
		title=(
			"Mixed SC / BCC crystal lattice with nested graph particles "
			f"({cells_per_axis} x {cells_per_axis} x {cells_per_axis})"
		),
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Crystal lattice"},
		hovermode="closest",
		scene={
			"xaxis": {"title": "X", "range": [-0.25, cells_per_axis + 0.25]},
			"yaxis": {"title": "Y", "range": [-0.25, cells_per_axis + 0.25]},
			"zaxis": {"title": "Z", "range": [-0.25, cells_per_axis + 0.25]},
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
