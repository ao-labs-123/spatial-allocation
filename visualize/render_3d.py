import json
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


def _axis_ticks(nodes, positions, attribute, axis_index, maximum_ticks=8):
	labels_by_key = {}
	values_by_key = {}
	for node in nodes:
		label = _axis_label(node, attribute)
		key = label.casefold()
		labels_by_key.setdefault(key, label)
		values_by_key.setdefault(key, []).append(positions[node["id"]][axis_index])

	ticks = sorted(
		((sum(values) / len(values), labels_by_key[key]) for key, values in values_by_key.items()),
		key=lambda tick: tick[0],
	)
	if len(ticks) > maximum_ticks:
		indices = {
			round(index * (len(ticks) - 1) / (maximum_ticks - 1))
			for index in range(maximum_ticks)
		}
		ticks = [tick for index, tick in enumerate(ticks) if index in indices]
	return [tick[0] for tick in ticks], [tick[1] for tick in ticks]


def render_graph(graph_data, output_path):
	nodes = graph_data["nodes"]
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

	for category in categories:
		category_nodes = [node for node in nodes if node.get("category", "Unknown") == category]
		figure.add_trace(
			go.Scatter3d(
				x=[positions[node["id"]][0] for node in category_nodes],
				y=[positions[node["id"]][1] for node in category_nodes],
				z=[positions[node["id"]][2] for node in category_nodes],
				mode="markers",
				name=category,
				text=[node.get("label", node["id"]) for node in category_nodes],
				marker={
					"size": 6,
					"color": category_colors[category],
					"line": {"color": "white", "width": 0.7},
				},
				customdata=[
					[
						node["id"],
						_axis_label(node, "what"),
						_axis_label(node, "who"),
						_axis_label(node, "when"),
						_axis_label(node, "where"),
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
					+ "<br>Causal / hierarchy depth: %{y}"
					+ "<br>Process stage: %{z}<extra></extra>"
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

	x_ticks, x_labels = _axis_ticks(nodes, positions, "what", 0)
	z_values = sorted({point[2] for point in coordinates})
	figure.update_layout(
		title="Semantic Space Allocation",
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Category"},
		hovermode="closest",
		scene={
			"xaxis": {
				"title": "What (lexical similarity, 1D)",
				"range": axis_ranges[0],
				"tickmode": "array",
				"tickvals": x_ticks,
				"ticktext": x_labels,
			},
			"yaxis": {
				"title": "Causal / hierarchical depth",
				"range": axis_ranges[1],
				"dtick": 1,
			},
			"zaxis": {
				"title": "Process stage (Temporal relations / explicit stage)",
				"range": axis_ranges[2],
				"tickmode": "array",
				"tickvals": z_values,
				"ticktext": [f"Stage {value:g}" for value in z_values],
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
  
