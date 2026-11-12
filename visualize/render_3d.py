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
SITE_CAPSULE_RADIUS = 0.28
NESTED_PARTICLE_RADIUS = 0.16


def _site_key(kind, coordinate):
	return kind + ":" + ",".join(f"{value:g}" for value in coordinate)


def _site_capsule_trace(
	sites,
	populated_sites,
	name,
	color,
	radius=SITE_CAPSULE_RADIUS,
):
	x_values = []
	y_values = []
	z_values = []
	face_i = []
	face_j = []
	face_k = []
	face_customdata = []
	face_text = []
	latitude_steps = 8
	longitude_steps = 12

	for kind, center in sites:
		key = _site_key(kind, center)
		base = len(x_values)
		x_values.append(center[0])
		y_values.append(center[1])
		z_values.append(center[2] - radius)
		rings = []
		for latitude_index in range(1, latitude_steps):
			latitude = -math.pi / 2 + latitude_index * math.pi / latitude_steps
			ring = []
			for longitude_index in range(longitude_steps):
				longitude = longitude_index * math.tau / longitude_steps
				ring.append(len(x_values))
				x_values.append(
					center[0]
					+ radius * math.cos(latitude) * math.cos(longitude)
				)
				y_values.append(
					center[1]
					+ radius * math.cos(latitude) * math.sin(longitude)
				)
				z_values.append(center[2] + radius * math.sin(latitude))
			rings.append(ring)
		top = len(x_values)
		x_values.append(center[0])
		y_values.append(center[1])
		z_values.append(center[2] + radius)
		face_count = 0

		for longitude_index in range(longitude_steps):
			next_longitude = (longitude_index + 1) % longitude_steps
			face_i.append(base)
			face_j.append(rings[0][next_longitude])
			face_k.append(rings[0][longitude_index])
			for ring_index in range(len(rings) - 1):
				lower = rings[ring_index]
				upper = rings[ring_index + 1]
				face_i.extend(
					[lower[longitude_index], lower[next_longitude]]
				)
				face_j.extend(
					[lower[next_longitude], upper[next_longitude]]
				)
				face_k.extend(
					[upper[longitude_index], upper[longitude_index]]
				)
			face_i.append(top)
			face_j.append(rings[-1][longitude_index])
			face_k.append(rings[-1][next_longitude])
			face_count += 2 + 2 * (len(rings) - 1)

		populated_count = len(populated_sites.get(key, []))
		hover_text = f"{name}<br>{populated_count} nested graph particles"
		face_customdata.extend([key] * face_count)
		face_text.extend([hover_text] * face_count)

	return go.Mesh3d(
		x=x_values,
		y=y_values,
		z=z_values,
		i=face_i,
		j=face_j,
		k=face_k,
		customdata=face_customdata,
		text=face_text,
		hovertemplate="%{text}<extra></extra>",
		color=color,
		opacity=0.28,
		flatshading=False,
		lighting={"ambient": 0.7, "diffuse": 0.5, "specular": 0.25, "roughness": 0.65},
		lightposition={"x": 100, "y": 200, "z": 300},
		name=name,
	)


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
	site_graph_data = {}
	for key, site_nodes in all_sites.items():
		if not site_nodes:
			continue
		site_node_ids = {node["id"] for node in site_nodes}
		site_edges = [
			edge
			for edge in edges
			if edge["source"] in site_node_ids or edge["target"] in site_node_ids
		]
		label_by_id = {
			node["id"]: str(node.get("label", node["id"]))
			for node in nodes_by_id.values()
		}
		graph_node_ids = set(site_node_ids)
		for edge in site_edges:
			graph_node_ids.update((edge["source"], edge["target"]))
		center = tuple(
			float(value)
			for value in key.split(":", 1)[1].split(",")
		)
		graph_nodes = [
			nodes_by_id[node_id]
			for node_id in sorted(graph_node_ids)
		]
		node_count = len(graph_nodes)
		node_positions_in_capsule = {}
		for index, node in enumerate(graph_nodes):
			angle = index * math.pi * (3 - math.sqrt(5))
			z_unit = 1 - 2 * (index + 0.5) / node_count
			radial_unit = math.sqrt(max(0.0, 1 - z_unit * z_unit))
			node_positions_in_capsule[node["id"]] = [
				center[0] + NESTED_PARTICLE_RADIUS * radial_unit * math.cos(angle),
				center[1] + NESTED_PARTICLE_RADIUS * radial_unit * math.sin(angle),
				center[2] + NESTED_PARTICLE_RADIUS * z_unit,
			]
		site_graph_data[key] = {
			"kind": "BCC center" if key.startswith("center:") else "Shared lattice vertex",
			"nodes": [
				{
					"id": node["id"],
					"label": str(node.get("label", node["id"])),
					"category": str(node.get("category", "")),
					"lattice_role": str(node.get("lattice_role", "")),
					"lattice_index": str(node.get("lattice_index", "")),
					"nested": node["id"] in site_node_ids,
					"position": node_positions_in_capsule[node["id"]],
				}
				for node in graph_nodes
			],
			"edges": [
				{
					"source": edge["source"],
					"target": edge["target"],
					"source_label": label_by_id[edge["source"]],
					"target_label": label_by_id[edge["target"]],
					"type": str(edge.get("morphism_type", "Relation")),
					"source_position": node_positions_in_capsule[edge["source"]],
					"target_position": node_positions_in_capsule[edge["target"]],
				}
				for edge in site_edges
			],
		}

	nested_edge_x = []
	nested_edge_y = []
	nested_edge_z = []
	nested_nodes = []
	for site_data in site_graph_data.values():
		for edge in site_data["edges"]:
			nested_edge_x.extend(
				[edge["source_position"][0], edge["target_position"][0], None]
			)
			nested_edge_y.extend(
				[edge["source_position"][1], edge["target_position"][1], None]
			)
			nested_edge_z.extend(
				[edge["source_position"][2], edge["target_position"][2], None]
			)
		nested_nodes.extend(site_data["nodes"])

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
		_site_capsule_trace(
			[("vertex", point) for point in lattice_vertices],
			vertex_contents,
			"Shared lattice vertices",
			"rgba(65, 145, 180, 1)",
		),
		_site_capsule_trace(
			[("center", point) for point in bcc_centers],
			center_contents,
			"BCC centers",
			"rgba(220, 140, 55, 1)",
		),
		go.Scatter3d(
			x=nested_edge_x,
			y=nested_edge_y,
			z=nested_edge_z,
			mode="lines",
			line={"color": "rgba(47, 55, 64, 0.95)", "width": 5},
			hoverinfo="skip",
			showlegend=False,
			name="Nested topology edges",
		),
		go.Scatter3d(
			x=[node["position"][0] for node in nested_nodes],
			y=[node["position"][1] for node in nested_nodes],
			z=[node["position"][2] for node in nested_nodes],
			mode="markers",
			marker={
				"size": 7,
				"line": {"color": "white", "width": 1},
				"color": [
					"#4477aa" if node["category"].lower() == "entity" else "#cc6677"
					for node in nested_nodes
				],
			},
			text=[node["label"] for node in nested_nodes],
			customdata=[
				[node["id"], node["category"], node["lattice_role"], node["lattice_index"]]
				for node in nested_nodes
			],
			hovertemplate=(
				"<b>%{text}</b><br>Category: %{customdata[1]}"
				+ "<br>ID: %{customdata[0]}"
				+ "<br>Lattice role: %{customdata[2]}"
				+ "<br>Index label: %{customdata[3]}<extra></extra>"
			),
			showlegend=False,
			name="Nested topology particles",
		),
	]

	return {
		"traces": geometry,
		"site_graph_data": site_graph_data,
		"cell_groups": cell_groups,
		"bcc_cells": bcc_cells,
		"cells_per_axis": cells_per_axis,
		"simple_cubic_bonds": geometry[0],
		"bcc_bonds": geometry[1],
		"vertices": geometry[2],
		"centers": geometry[3],
		"site_contents": all_sites,
		"nested_nodes": nested_nodes,
		"nested_edges": geometry[4],
		"nested_particles": geometry[5],
	}


def render_lattice(graph_data, output_path, seed=None):
	geometry = _mixed_lattice_geometry(graph_data, seed)
	figure = go.Figure(data=geometry["traces"])

	cells_per_axis = geometry["cells_per_axis"]
	figure.update_layout(
		title=(
			"Mixed SC / BCC crystal lattice with visible nested topology "
			f"({cells_per_axis} x {cells_per_axis} x {cells_per_axis})"
		),
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Crystal lattice"},
		hovermode="closest",
		scene={
			"xaxis": {"title": "X", "range": [-0.5, cells_per_axis + 0.5]},
			"yaxis": {"title": "Y", "range": [-0.5, cells_per_axis + 0.5]},
			"zaxis": {"title": "Z", "range": [-0.5, cells_per_axis + 0.5]},
			"aspectmode": "data",
			"dragmode": "orbit",
			"camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
		},
	)
	output_path = Path(output_path)
	figure.write_html(
		output_path,
		include_plotlyjs=True,
		full_html=True,
	)
	if output_path.exists():
		html_lines = output_path.read_text(encoding="utf-8").splitlines()
		output_path.write_text(
			"\n".join(line.rstrip() for line in html_lines) + "\n",
			encoding="utf-8",
		)


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph.json"
	output_path = project_dir / "docs" / "index.html"
	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	render_lattice(graph_data, output_path)
	print(f"Saved mixed crystal lattice visualization to {output_path}")


if __name__ == "__main__":
	main()
