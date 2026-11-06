import json
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
	for container in containers:
		center = positions[container["id"]]
		radius = float(container["capsule_radius"])
		half_length = float(container["capsule_half_length"])

		for x_offset in (-half_length, 0.0, half_length):
			for index in range(25):
				angle = 2 * math.pi * index / 24
				x_values.append(center[0] + x_offset)
				y_values.append(center[1] + radius * math.cos(angle))
				z_values.append(center[2] + radius * math.sin(angle))
			x_values.append(None)
			y_values.append(None)
			z_values.append(None)

		for index in range(12):
			angle = 2 * math.pi * index / 12
			for side in (-1, 1):
				for step in range(13):
					arc = (math.pi / 2) * step / 12
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


def render_graph(graph_data, output_path):
	nodes = graph_data["nodes"]
	leaf_nodes = [node for node in nodes if node.get("node_type") != "container"]
	containers = [node for node in nodes if node.get("node_type") == "container"]
	positions = {}
	for node in nodes:
		position = node.get("position")
		if not isinstance(position, list) or len(position) != 3:
			raise ValueError(f"Node {node.get('id', '<unknown>')} has no valid 3D position")
		positions[node["id"]] = position

	categories = sorted({node.get("category", "Unknown") for node in nodes})
	palette = qualitative.Safe + qualitative.Vivid + qualitative.Pastel
	category_colors = {category: palette[index % len(palette)] for index, category in enumerate(categories)}
	figure = go.Figure()
	cluster_containers = [container for container in containers if container.get("container_type") != "conversation"]
	if cluster_containers:
		figure.add_trace(_container_capsule_trace(cluster_containers, positions))

	edge_x = []
	edge_y = []
	edge_z = []
	for edge in graph_data["edges"]:
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
				x=[positions[node["id"]][0] for node in category_nodes],
				y=[positions[node["id"]][1] for node in category_nodes],
				z=[positions[node["id"]][2] for node in category_nodes],
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
					+ "<br>Causal / hierarchy depth: %{y}"
					+ "<br>Process stage: %{z}<extra></extra>"
				),
			)
		)

	for container_type, label, symbol, size in (
		("cluster", "Nested containers", "diamond", 11),
		("conversation", "Conversation container", "diamond-open", 15),
	):
		container_nodes = [container for container in containers if container.get("container_type") == container_type]
		if not container_nodes:
			continue
		figure.add_trace(
			go.Scatter3d(
				x=[positions[node["id"]][0] for node in container_nodes],
				y=[positions[node["id"]][1] for node in container_nodes],
				z=[positions[node["id"]][2] for node in container_nodes],
				mode="markers+text",
				name=label,
				text=[node.get("label", node["id"]) for node in container_nodes],
				textposition="top center",
				customdata=[
					[
						node["id"],
						len(node.get("contains", [])),
						node.get("parent_id", "Top level"),
						*(node.get("semantic_position") or [0.0, 0.0, 0.0]),
						node.get("lattice_mode", ""),
						", ".join(
							f"{core['label']} ({core['node_id']})"
							for core in node.get("core_indices", [])
						),
					]
					for node in container_nodes
				],
				marker={
					"size": size,
					"symbol": symbol,
					"color": "#245b78",
					"line": {"color": "white", "width": 1.2},
				},
				hovertemplate=(
					"<b>%{text}</b><br>Container particle"
					+ "<br>ID: %{customdata[0]}"
					+ "<br>Contains: %{customdata[1]} particles"
					+ "<br>Parent: %{customdata[2]}"
					+ "<br>Semantic center: (%{customdata[3]:.2f}, %{customdata[4]:.2f}, %{customdata[5]:.2f})"
					+ "<br>Lattice: %{customdata[6]}"
					+ "<br>Core indices: %{customdata[7]}"
					+ "<extra></extra>"
				),
			)
		)

	coordinates = list(positions.values())
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
		title="Semantic Space Allocation",
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Category"},
		hovermode="closest",
		scene={
			"xaxis": {
				"title": "Packed container layout X",
				"range": axis_ranges[0],
			},
			"yaxis": {
				"title": "Packed container layout Y",
				"range": axis_ranges[1],
			},
			"zaxis": {
				"title": "Packed container layout Z",
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
  
