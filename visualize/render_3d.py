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


def _particle_marker_color(node, category_color):
	if (
		str(node.get("category", "")).casefold() == "event"
		and node.get("lattice_role") != "vertex"
	):
		return "rgba(145, 153, 160, 0.9)"
	return category_color


def _crystal_geometry(crystal):
	repeats = crystal.get("preview_repeats", [1, 1, 1])
	unit_cell = crystal.get("unit_cell")
	if (
		not isinstance(repeats, list)
		or len(repeats) != 3
		or any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in repeats)
	):
		raise ValueError("Crystal preview_repeats must contain three positive integers")
	if unit_cell is None:
		return None, None

	center = unit_cell.get("center")
	size = unit_cell.get("size")
	if (
		not isinstance(center, list)
		or len(center) != 3
		or not isinstance(size, list)
		or len(size) != 3
		or any(not isinstance(value, (int, float)) for value in center + size)
		or any(value <= 0 for value in size)
	):
		raise ValueError("Crystal unit_cell must contain a 3D center and positive size")

	cell_ranges = [
		range(-(count // 2), count - count // 2)
		for count in repeats
	]
	cells = list(itertools.product(*cell_ranges))
	vertices = {
		tuple(cell[axis] + side[axis] * 0.5 for axis in range(3))
		for cell in cells
		for side in itertools.product((-1, 1), repeat=3)
	}
	body_centers = set(cells) if crystal.get("lattice_mode") == "body_centered_cubic" else set()

	x_values = []
	y_values = []
	z_values = []
	if not body_centers:
		for vertex in sorted(vertices):
			for axis in range(3):
				neighbor = list(vertex)
				neighbor[axis] += 1
				neighbor = tuple(neighbor)
				if neighbor not in vertices:
					continue
				left = [center[index] + vertex[index] * size[index] for index in range(3)]
				right = [center[index] + neighbor[index] * size[index] for index in range(3)]
				x_values.extend([left[0], right[0], None])
				y_values.extend([left[1], right[1], None])
				z_values.extend([left[2], right[2], None])

	if body_centers:
		for cell in sorted(body_centers):
			body_position = [center[axis] + cell[axis] * size[axis] for axis in range(3)]
			for corner in itertools.product((-0.5, 0.5), repeat=3):
				vertex = tuple(cell[axis] + corner[axis] for axis in range(3))
				if vertex not in vertices:
					continue
				vertex_position = [center[axis] + vertex[axis] * size[axis] for axis in range(3)]
				x_values.extend([body_position[0], vertex_position[0], None])
				y_values.extend([body_position[1], vertex_position[1], None])
				z_values.extend([body_position[2], vertex_position[2], None])

	bond_trace = go.Scatter3d(
		x=x_values,
		y=y_values,
		z=z_values,
		mode="lines",
		line={"color": "rgba(35, 40, 45, 0.58)", "width": 3},
		hoverinfo="skip",
		showlegend=True,
		name=f"Outer {crystal.get('lattice_mode', 'simple_cubic')} lattice",
	)
	lattice_sites = sorted(vertices | body_centers)
	site_positions = [
		[center[axis] + site[axis] * size[axis] for axis in range(3)]
		for site in lattice_sites
	]
	site_trace = go.Scatter3d(
		x=[position[0] for position in site_positions],
		y=[position[1] for position in site_positions],
		z=[position[2] for position in site_positions],
		mode="markers",
		marker={
			"size": 4,
			"color": "rgba(100, 110, 120, 0.55)",
			"symbol": [
				"diamond" if site in body_centers else "circle"
				for site in lattice_sites
			],
		},
		hovertemplate="Outer lattice site<extra></extra>",
		showlegend=False,
		name="Outer lattice sites",
	)
	return bond_trace, site_trace


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


def _topology_container_trace(containers, positions):
	x_values = []
	y_values = []
	z_values = []
	ring_segments = 16
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
		line={"color": "rgba(36, 91, 120, 0.7)", "width": 2},
		hoverinfo="skip",
		showlegend=True,
		name="Topology containers",
	)


def _inner_lattice_geometry(containers, particles, positions):
	container_by_cluster_and_cell = {
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
		container = container_by_cluster_and_cell.get((cluster_id, cell))
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
		body_centers = set()
		if container.get("lattice_mode") == "body_centered_cubic":
			body_centers = {
				(x + 0.5, y + 0.5, z + 0.5)
				for x in range(lower[0], upper[0])
				for y in range(lower[1], upper[1])
				for z in range(lower[2], upper[2])
			}
		sites = corners | body_centers
		site_positions = {site: lattice_position(site) for site in sites}
		for site in corners:
			position = site_positions[site]
			vertex_x.append(position[0])
			vertex_y.append(position[1])
			vertex_z.append(position[2])
			vertex_coordinates.append(
				"Lattice vertex: " + ", ".join(f"{value:g}" for value in site)
			)
		for site in body_centers:
			position = site_positions[site]
			center_x.append(position[0])
			center_y.append(position[1])
			center_z.append(position[2])
			center_coordinates.append(
				"BCC center: " + ", ".join(f"{value:g}" for value in site)
			)

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
			"size": 14,
			"color": "rgba(145, 153, 160, 0.55)",
			"line": {"color": "white", "width": 0.8},
		},
		customdata=vertex_coordinates,
		hovertemplate="%{customdata}<extra></extra>",
		name="Lattice vertices",
		showlegend=bool(vertex_x),
	)
	center_trace = go.Scatter3d(
		x=center_x,
		y=center_y,
		z=center_z,
		mode="markers",
		marker={
			"size": 14,
			"symbol": "diamond",
			"color": "rgba(145, 153, 160, 0.55)",
			"line": {"color": "white", "width": 0.8},
		},
		customdata=center_coordinates,
		hovertemplate="%{customdata}<extra></extra>",
		name="BCC center sites",
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
	outer_geometry = _crystal_geometry(crystal)
	repeats = crystal.get("preview_repeats", [1, 1, 1])
	unit_cell = crystal.get("unit_cell") or {"size": [0.0, 0.0, 0.0]}
	cell_ranges = [
		range(-(count // 2), count - count // 2)
		for count in repeats
	]
	cells = list(itertools.product(*cell_ranges))
	rendered_nodes = []
	rendered_edges = []
	positions = {}
	for cell in cells:
		offset = [cell[axis] * unit_cell["size"][axis] for axis in range(3)]
		render_id_by_source_id = {
			node["id"]: f"{node['id']}@{cell[0]},{cell[1]},{cell[2]}"
			for node in source_nodes
			if node.get("container_type") != "conversation"
		}
		for node in source_nodes:
			if node.get("container_type") == "conversation":
				continue
			render_id = render_id_by_source_id[node["id"]]
			rendered_node = dict(node)
			rendered_node["_render_id"] = render_id
			rendered_node["_render_parent_id"] = render_id_by_source_id.get(node.get("parent_id"))
			rendered_node["topology_cluster_id"] = node.get("parent_id")
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
	all_containers = containers

	categories = sorted({node.get("category", "Unknown") for node in leaf_nodes})
	palette = qualitative.Safe + qualitative.Vivid + qualitative.Pastel
	category_colors = {category: palette[index % len(palette)] for index, category in enumerate(categories)}
	figure = go.Figure()
	if outer_geometry[0] is not None:
		figure.add_trace(outer_geometry[1])
	if containers:
		figure.add_trace(_topology_container_trace(containers, positions))
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
					"color": [
						_particle_marker_color(node, category_colors[category])
						for node in category_nodes
					],
					"line": {"color": "white", "width": 0.7},
				},
				customdata=[
					[
						node.get("source_id", node["id"]),
						_axis_label(node, "what"),
						_axis_label(node, "who"),
						_axis_label(node, "when"),
						_axis_label(node, "where"),
						(
							"Unassigned"
							if str(node.get("category", "")).casefold() == "event"
							and node.get("lattice_role") != "vertex"
							else node.get("lattice_role", "particle")
						),
						node.get("lattice_index", ""),
						node.get("lattice_site", ""),
						", ".join(str(value) for value in node.get("lattice_coordinate", [])),
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
					+ "<br>Lattice assignment: %{customdata[5]}"
					+ "<br>Index: %{customdata[6]}"
					+ "<br>Lattice site: %{customdata[7]}"
					+ "<br>Lattice coordinate: %{customdata[8]}"
					+ "<br>Causal / hierarchy depth: %{y}"
					+ "<br>Process stage: %{z}<extra></extra>"
				),
			)
		)

	container_nodes = [
		container for container in all_containers
		if container.get("container_type") == "cluster"
	]
	if container_nodes:
		figure.add_trace(
			go.Scatter3d(
				x=[positions[node["_render_id"]][0] for node in container_nodes],
				y=[positions[node["_render_id"]][1] for node in container_nodes],
				z=[
					positions[node["_render_id"]][2] + float(node["capsule_radius"]) + 0.15
					for node in container_nodes
				],
				mode="text",
				name="Topology container labels",
				text=[node.get("label", node["id"]) for node in container_nodes],
				textposition="top center",
				customdata=[
					[
						node["id"],
						len(node.get("contains", [])),
						node.get("parent_id", "Top level"),
						*(node.get("semantic_position") or [0.0, 0.0, 0.0]),
						node.get("lattice_mode", ""),
						", ".join(core["label"] for core in node.get("core_indices", [])),
					]
					for node in container_nodes
				],
				hovertemplate=(
					"<b>%{text}</b><br>Container particle"
					+ "<br>ID: %{customdata[0]}"
					+ "<br>Contains: %{customdata[1]} particles"
					+ "<br>Parent: %{customdata[2]}"
					+ "<br>Semantic center: (%{customdata[3]:.2f}, %{customdata[4]:.2f}, %{customdata[5]:.2f})"
					+ "<br>Inner lattice: %{customdata[6]}"
					+ "<br>Core indices: %{customdata[7]}"
					+ "<extra></extra>"
				),
			)
		)

	coordinates = list(positions.values())
	for container in containers:
		center = positions[container["_render_id"]]
		radius = float(container["capsule_radius"])
		half_length = float(container["capsule_half_length"])
		coordinates.extend(
			[
				center[0] + x_side * (half_length + radius),
				center[1] + y_side * radius,
				center[2] + z_side * radius,
			]
			for x_side, y_side, z_side in itertools.product((-1, 1), repeat=3)
		)
		coordinates.append([center[0], center[1], center[2] + radius + 0.15])
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
		title="Semantic Topology in a Periodic Crystal Lattice",
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Category"},
		hovermode="closest",
		scene={
			"xaxis": {
				"title": "Crystal lattice X",
				"range": axis_ranges[0],
			},
			"yaxis": {
				"title": "Crystal lattice Y",
				"range": axis_ranges[1],
			},
			"zaxis": {
				"title": "Crystal lattice Z",
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
  
