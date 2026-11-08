import json
import itertools
import math
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import qualitative


def _axis_label(node, attribute):
	attributes = node.get("attributes") or {}
	value = attributes.get(attribute)
	if value is None or not str(value).strip():
		return "Unspecified"
	value = str(value).strip()
	return "Unspecified" if value.casefold() == "unspecified" else value


def _container_capsule_trace(containers, positions):
	x_values = []
	y_values = []
	z_values = []
	ring_segments = 12
	cap_segments = 8
	arc_steps = 5
	for container in containers:
		center = positions[container["_render_id"]]
		radius = float(container["capsule_radius"])
		half_length = float(container["capsule_half_length"])

		for x_offset in (-half_length, 0.0, half_length):
			for index in range(ring_segments + 1):
				angle = 2 * math.pi * index / ring_segments
				x_values.append(center[0] + x_offset)
				y_values.append(center[1] + radius * math.cos(angle))
				z_values.append(center[2] + radius * math.sin(angle))
			x_values.append(None)
			y_values.append(None)
			z_values.append(None)

		for index in range(cap_segments):
			angle = 2 * math.pi * index / cap_segments
			for side in (-1, 1):
				for step in range(arc_steps + 1):
					arc = (math.pi / 2) * step / arc_steps
					x_values.append(center[0] + side * (half_length + radius * math.sin(arc)))
					radial = radius * math.cos(arc)
					y_values.append(center[1] + radial * math.cos(angle))
					z_values.append(center[2] + radial * math.sin(angle))
				x_values.append(None)
				y_values.append(None)
				z_values.append(None)
	return go.Scatter3d(
		x=x_values,
		y=y_values,
		z=z_values,
		mode="lines",
		line={"color": "rgba(55, 110, 140, 0.55)", "width": 2},
		hoverinfo="skip",
		showlegend=False,
		name="Capsule containers",
	)


def _crystal_cells(crystal):
	repeats = crystal.get("preview_repeats", [1, 1, 1])
	unit_cell = crystal.get("unit_cell")
	if (
		not isinstance(repeats, list)
		or len(repeats) != 3
		or any(not isinstance(count, int) or count < 1 for count in repeats)
	):
		raise ValueError("Crystal preview_repeats must contain three positive integers")
	if unit_cell is not None:
		size = unit_cell.get("size")
		center = unit_cell.get("center")
		if (
			not isinstance(size, list)
			or len(size) != 3
			or not isinstance(center, list)
			or len(center) != 3
		):
			raise ValueError("Crystal unit_cell must contain 3D center and size values")
	else:
		size = [0.0, 0.0, 0.0]
		center = [0.0, 0.0, 0.0]

	ranges = [
		range(-(count // 2), count - count // 2)
		for count in repeats
	]
	return list(itertools.product(*ranges)), center, size


def _unit_cell_trace(crystal, cell_coordinates):
	unit_cell = crystal.get("unit_cell")
	if unit_cell is None:
		return None

	center = unit_cell["center"]
	size = unit_cell["size"]
	x_values = []
	y_values = []
	z_values = []
	for cell in cell_coordinates:
		cell_center = [
			center[axis] + cell[axis] * size[axis]
			for axis in range(3)
		]
		corners = {
			(x_side, y_side, z_side): [
				cell_center[0] + x_side * size[0] / 2,
				cell_center[1] + y_side * size[1] / 2,
				cell_center[2] + z_side * size[2] / 2,
			]
			for x_side, y_side, z_side in itertools.product((-1, 1), repeat=3)
		}
		for corner, position in corners.items():
			for axis in range(3):
				neighbor = list(corner)
				neighbor[axis] *= -1
				neighbor = tuple(neighbor)
				if corner[axis] != -1 or neighbor not in corners:
					continue
				other = corners[neighbor]
				x_values.extend([position[0], other[0], None])
				y_values.extend([position[1], other[1], None])
				z_values.extend([position[2], other[2], None])
	return go.Scatter3d(
		x=x_values,
		y=y_values,
		z=z_values,
		mode="lines",
		line={"color": "rgba(75, 80, 86, 0.78)", "width": 2},
		hoverinfo="skip",
		showlegend=True,
		name="Outer supercell boundaries (unit cells)",
	)


def _outer_lattice_bond_trace(crystal, cell_coordinates):
	unit_cell = crystal.get("unit_cell")
	if unit_cell is None:
		return None

	center = unit_cell["center"]
	size = unit_cell["size"]
	cell_centers = {
		tuple(cell): [
			center[axis] + cell[axis] * size[axis]
			for axis in range(3)
		]
		for cell in cell_coordinates
	}
	x_values = []
	y_values = []
	z_values = []
	for cell, position in cell_centers.items():
		for axis in range(3):
			neighbor = list(cell)
			neighbor[axis] += 1
			neighbor_position = cell_centers.get(tuple(neighbor))
			if neighbor_position is None:
				continue
			x_values.extend([position[0], neighbor_position[0], None])
			y_values.extend([position[1], neighbor_position[1], None])
			z_values.extend([position[2], neighbor_position[2], None])

	return go.Scatter3d(
		x=x_values,
		y=y_values,
		z=z_values,
		mode="lines",
		line={"color": "rgba(35, 40, 45, 0.62)", "width": 4},
		hoverinfo="skip",
		showlegend=True,
		name="Supercell repeat connections",
	)


def _inner_lattice_geometry(containers, particles, positions):
	containers_by_cluster_and_cell = {
		(container["id"], tuple(container["cell_coordinate"])): container
		for container in containers
	}
	particles_by_cluster_and_cell = {}
	for node in particles:
		if isinstance(node.get("lattice_coordinate"), list):
			key = (node.get("topology_cluster_id"), tuple(node["cell_coordinate"]))
			particles_by_cluster_and_cell.setdefault(key, []).append(node)

	x_values = []
	y_values = []
	z_values = []
	vertex_x = []
	vertex_y = []
	vertex_z = []
	vertex_coordinates = []
	center_x = []
	center_y = []
	center_z = []
	center_coordinates = []

	for (cluster_id, cell), group in particles_by_cluster_and_cell.items():
		container = containers_by_cluster_and_cell.get((cluster_id, cell))
		if container is None:
			continue

		lattice_coordinates = [
			tuple(float(value) for value in node["lattice_coordinate"])
			for node in group
		]
		lower = []
		upper = []
		for axis in range(3):
			minimum = min(point[axis] for point in lattice_coordinates)
			maximum = max(point[axis] for point in lattice_coordinates)
			axis_lower = math.floor(minimum + 1e-9)
			axis_upper = math.ceil(maximum - 1e-9)
			if axis_upper <= axis_lower:
				axis_upper = axis_lower + 1
			lower.append(axis_lower)
			upper.append(axis_upper)

		container_center = positions[container["_render_id"]]
		target_extents = [
			float(container["capsule_half_length"]) * 0.72,
			float(container["capsule_radius"]) * 0.62,
			float(container["capsule_radius"]) * 0.62,
		]

		def lattice_position(coordinate):
			return [
				container_center[axis]
				+ (
					(coordinate[axis] - (lower[axis] + upper[axis]) / 2)
					/ (upper[axis] - lower[axis])
					* 2
					* target_extents[axis]
				)
				for axis in range(3)
			]

		for node, coordinate in zip(group, lattice_coordinates):
			positions[node["_render_id"]] = lattice_position(coordinate)

		corners = {
			(x, y, z)
			for x in range(lower[0], upper[0] + 1)
			for y in range(lower[1], upper[1] + 1)
			for z in range(lower[2], upper[2] + 1)
		}
		sites = set(corners)
		body_centers = set()
		if container.get("lattice_mode") == "body_centered_cubic":
			body_centers = {
				(x + 0.5, y + 0.5, z + 0.5)
				for x in range(lower[0], upper[0])
				for y in range(lower[1], upper[1])
				for z in range(lower[2], upper[2])
			}
			sites.update(body_centers)

		site_positions = {site: lattice_position(site) for site in sites}
		occupied = {
			tuple(round(value, 8) for value in coordinate)
			for coordinate in lattice_coordinates
		}
		for site in corners:
			if site in occupied:
				continue
			position = site_positions[site]
			vertex_x.append(position[0])
			vertex_y.append(position[1])
			vertex_z.append(position[2])
			vertex_coordinates.append(", ".join(f"{value:g}" for value in site))
		for site in body_centers:
			if site in occupied:
				continue
			position = site_positions[site]
			center_x.append(position[0])
			center_y.append(position[1])
			center_z.append(position[2])
			center_coordinates.append(", ".join(f"{value:g}" for value in site))

		for site in corners:
			for axis in range(3):
				neighbor = list(site)
				neighbor[axis] += 1
				neighbor = tuple(neighbor)
				if neighbor not in corners:
					continue
				left_position = site_positions[site]
				right_position = site_positions[neighbor]
				x_values.extend([left_position[0], right_position[0], None])
				y_values.extend([left_position[1], right_position[1], None])
				z_values.extend([left_position[2], right_position[2], None])

		for site in body_centers:
			cell_origin = tuple(math.floor(value) for value in site)
			for corner in itertools.product((0, 1), repeat=3):
				vertex = tuple(cell_origin[axis] + corner[axis] for axis in range(3))
				left_position = site_positions[site]
				right_position = site_positions[vertex]
				x_values.extend([left_position[0], right_position[0], None])
				y_values.extend([left_position[1], right_position[1], None])
				z_values.extend([left_position[2], right_position[2], None])

	bond_trace = go.Scatter3d(
		x=x_values,
		y=y_values,
		z=z_values,
		mode="lines",
		line={"color": "rgba(35, 40, 45, 0.92)", "width": 3},
		hoverinfo="skip",
		showlegend=True,
		name="Inner SC / BCC lattice bonds",
	)
	vertex_trace = go.Scatter3d(
		x=vertex_x,
		y=vertex_y,
		z=vertex_z,
		mode="markers",
		marker={
			"size": 6,
			"color": "rgba(145, 153, 160, 0.78)",
			"line": {"color": "white", "width": 0.8},
		},
		customdata=vertex_coordinates,
		hovertemplate="Unassigned lattice vertex<br>Coordinate: (%{customdata})<extra></extra>",
		name="Unassigned lattice vertices",
		showlegend=bool(vertex_x),
	)
	center_trace = go.Scatter3d(
		x=center_x,
		y=center_y,
		z=center_z,
		mode="markers",
		marker={
			"size": 7,
			"symbol": "diamond",
			"color": "rgba(110, 119, 127, 0.82)",
			"line": {"color": "white", "width": 0.8},
		},
		customdata=center_coordinates,
		hovertemplate="Unassigned BCC center site<br>Coordinate: (%{customdata})<extra></extra>",
		name="Unassigned BCC center sites",
		showlegend=bool(center_x),
	)
	return bond_trace, vertex_trace, center_trace


def render_graph(graph_data, output_path):
	source_nodes = graph_data["nodes"]
	source_positions = {}
	for node in source_nodes:
		position = node.get("position")
		if not isinstance(position, list) or len(position) != 3:
			raise ValueError(f"Node {node.get('id', '<unknown>')} has no valid 3D position")
		source_positions[node["id"]] = position

	crystal = graph_data.get("crystal") or {}
	cell_coordinates, _, cell_size = _crystal_cells(crystal)
	rendered_nodes = []
	positions = {}
	rendered_edges = []
	for cell in cell_coordinates:
		offset = [cell[axis] * cell_size[axis] for axis in range(3)]
		render_id_by_source_id = {}
		for node in source_nodes:
			if node.get("container_type") == "conversation":
				continue
			render_id = f"{node['id']}@{cell[0]},{cell[1]},{cell[2]}"
			render_id_by_source_id[node["id"]] = render_id
			rendered_node = dict(node)
			rendered_node["_render_id"] = render_id
			rendered_node["cell_coordinate"] = list(cell)
			rendered_node["position"] = [
				float(source_positions[node["id"]][axis]) + offset[axis]
				for axis in range(3)
			]
			if node.get("container_type") == "cluster" and any(cell):
				rendered_node["label"] = ""
			positions[render_id] = rendered_node["position"]
			rendered_nodes.append(rendered_node)
		for edge in graph_data["edges"]:
			source = render_id_by_source_id.get(edge["source"])
			target = render_id_by_source_id.get(edge["target"])
			if source is not None and target is not None:
				rendered_edges.append({"source": source, "target": target})

	leaf_nodes = [node for node in rendered_nodes if node.get("node_type") != "container"]
	containers = [node for node in rendered_nodes if node.get("container_type") == "cluster"]

	categories = sorted({node.get("category", "Unknown") for node in leaf_nodes})
	palette = qualitative.Safe + qualitative.Vivid + qualitative.Pastel
	category_colors = {category: palette[index % len(palette)] for index, category in enumerate(categories)}
	figure = go.Figure()
	unit_cells = _unit_cell_trace(crystal, cell_coordinates)
	if unit_cells is not None:
		figure.add_trace(unit_cells)
	outer_lattice_bonds = _outer_lattice_bond_trace(crystal, cell_coordinates)
	if outer_lattice_bonds is not None and outer_lattice_bonds.x:
		figure.add_trace(outer_lattice_bonds)
	if containers:
		figure.add_trace(_container_capsule_trace(containers, positions))
	lattice_bonds, unassigned_vertices, unassigned_centers = _inner_lattice_geometry(
		containers,
		leaf_nodes,
		positions,
	)
	if lattice_bonds.x:
		figure.add_trace(lattice_bonds)
	if unassigned_vertices.x:
		figure.add_trace(unassigned_vertices)
	if unassigned_centers.x:
		figure.add_trace(unassigned_centers)

	edge_x = []
	edge_y = []
	edge_z = []
	for edge in rendered_edges:
		source = positions.get(edge["source"])
		target = positions.get(edge["target"])
		if source is None or target is None:
			continue
		edge_x.extend([source[0], target[0], None])
		edge_y.extend([source[1], target[1], None])
		edge_z.extend([source[2], target[2], None])

	figure.add_trace(
		go.Scatter3d(
			x=edge_x,
			y=edge_y,
			z=edge_z,
			mode="lines",
			line={"color": "rgba(98, 106, 115, 0.55)", "width": 3},
			hoverinfo="skip",
			showlegend=False,
			name="Edges",
		)
	)

	for category in sorted({node.get("category", "Unknown") for node in leaf_nodes}):
		category_nodes = [node for node in leaf_nodes if node.get("category", "Unknown") == category]
		figure.add_trace(
			go.Scatter3d(
				x=[positions[node["_render_id"]][0] for node in category_nodes],
				y=[positions[node["_render_id"]][1] for node in category_nodes],
				z=[positions[node["_render_id"]][2] for node in category_nodes],
				mode="markers",
				name=category,
				text=[node.get("label", node["id"]) for node in category_nodes],
				marker={
					"size": [
						9 if node.get("lattice_role") == "core" else 6
						for node in category_nodes
					],
					"color": category_colors[category],
					"symbol": [
						"diamond" if node.get("lattice_role") == "core" else "circle"
						for node in category_nodes
					],
					"line": {"color": "white", "width": 0.7},
				},
				customdata=[
					[
						node["id"],
						_axis_label(node, "what"),
						_axis_label(node, "who"),
						_axis_label(node, "when"),
						_axis_label(node, "where"),
						node.get("lattice_role", "particle"),
						node.get("lattice_index") or ", ".join(
							f"{core['label']} ({core['node_id']})"
							for core in node.get("lattice_cores", [])
						),
						node.get("lattice_site", ""),
						", ".join(f"{value:g}" for value in node.get("lattice_coordinate", [])),
						node.get("source_example_id", ""),
						node.get("source_text", ""),
						", ".join(str(value) for value in node["cell_coordinate"]),
					]
					for node in category_nodes
				],
				hovertemplate=(
					"<b>%{text}</b><br>Category: "
					+ category
					+ "<br>ID: %{customdata[0]}"
					+ "<br>What: %{customdata[1]}"
					+ "<br>Who: %{customdata[2]}"
					+ "<br>When: %{customdata[3]}"
					+ "<br>Where: %{customdata[4]}"
					+ "<br>Lattice role: %{customdata[5]}"
					+ "<br>Core index: %{customdata[6]}"
					+ "<br>Lattice site: %{customdata[7]}"
					+ "<br>Lattice coordinate: %{customdata[8]}"
					+ "<br>Source example: %{customdata[9]}"
					+ "<br>Input: %{customdata[10]}"
					+ "<br>Outer cell: (%{customdata[11]})"
					+ "<br>Causal / hierarchy depth: %{y}"
					+ "<br>Process stage: %{z}<extra></extra>"
				),
			)
		)

	if containers:
		figure.add_trace(
			go.Scatter3d(
				x=[positions[node["_render_id"]][0] for node in containers],
				y=[positions[node["_render_id"]][1] for node in containers],
				z=[positions[node["_render_id"]][2] for node in containers],
				mode="markers+text",
				name="Nested topology clusters",
				text=[node.get("label", node["id"]) for node in containers],
				textposition="top center",
				customdata=[
					[
						node["id"],
						len(node.get("contains", [])),
						node.get("lattice_mode", ""),
						", ".join(
							f"{core['label']} ({core['node_id']})"
							for core in node.get("core_indices", [])
						),
						", ".join(node.get("source_example_ids", [])),
						", ".join(str(value) for value in node["cell_coordinate"]),
					]
					for node in containers
				],
				marker={
					"size": 8,
					"symbol": "diamond",
					"color": "#245b78",
					"line": {"color": "white", "width": 1.2},
				},
				hovertemplate=(
					"<b>%{text}</b><br>Topology cluster"
					+ "<br>ID: %{customdata[0]}"
					+ "<br>Contains: %{customdata[1]} particles"
					+ "<br>Inner lattice: %{customdata[2]}"
					+ "<br>Core indices: %{customdata[3]}"
					+ "<br>Source examples: %{customdata[4]}"
					+ "<br>Outer cell: (%{customdata[5]})"
					+ "<extra></extra>"
				),
			)
		)

	coordinates = list(positions.values())
	unit_cell = crystal.get("unit_cell")
	if unit_cell is not None:
		for cell in cell_coordinates:
			cell_center = [
				unit_cell["center"][axis] + cell[axis] * unit_cell["size"][axis]
				for axis in range(3)
			]
			for corner in itertools.product((-1, 1), repeat=3):
				coordinates.append([
					cell_center[axis] + corner[axis] * unit_cell["size"][axis] / 2
					for axis in range(3)
				])
	axis_ranges = []
	for axis_index in range(3):
		axis_values = [point[axis_index] for point in coordinates]
		if not axis_values:
			axis_ranges.append([-1, 1])
			continue
		minimum = min(axis_values)
		maximum = max(axis_values)
		padding = max((maximum - minimum) * 0.08, 0.5)
		axis_ranges.append([minimum - padding, maximum + padding])

	figure.update_layout(
		title="Nested Topological Crystal (finite 3×3×3 preview)",
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Category"},
		hovermode="closest",
		scene={
			"xaxis": {
				"title": "Outer crystal lattice X",
				"range": axis_ranges[0],
			},
			"yaxis": {
				"title": "Outer crystal lattice Y",
				"range": axis_ranges[1],
			},
			"zaxis": {
				"title": "Outer crystal lattice Z",
				"range": axis_ranges[2],
			},
			"aspectmode": "data",
			"dragmode": "orbit",
			"camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
		},
	)
	figure.write_html(output_path, include_plotlyjs=True, full_html=True)


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph_3d.json"
	output_path = project_dir / "docs" / "index.html"

	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)

	output_path.parent.mkdir(parents=True, exist_ok=True)
	render_graph(graph_data, output_path)
	print(f"Saved interactive 3D visualization for GitHub Pages to {output_path}")


if __name__ == "__main__":
	main()
  
