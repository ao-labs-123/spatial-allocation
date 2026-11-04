import json
from pathlib import Path

import plotly.graph_objects as go
from plotly.colors import qualitative


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
				customdata=[node["id"] for node in category_nodes],
				marker={
					"size": 6,
					"color": category_colors[category],
					"line": {"color": "white", "width": 0.7},
				},
				hovertemplate=(
					"<b>%{text}</b><br>Category: "
					+ category
					+ "<br>ID: %{customdata}<extra></extra>"
				),
			)
		)

	coordinates = list(positions.values())
	center = [
		(min(point[axis_index] for point in coordinates) + max(point[axis_index] for point in coordinates)) / 2
		for axis_index in range(3)
	]
	radius = max(
		(max(point[axis_index] for point in coordinates) - min(point[axis_index] for point in coordinates)) / 2
		for axis_index in range(3)
	) or 1
	figure.update_layout(
		title="3D Topological Graph",
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Category"},
		hovermode="closest",
		scene={
			"xaxis": {"title": "X", "range": [center[0] - radius, center[0] + radius]},
			"yaxis": {"title": "Y", "range": [center[1] - radius, center[1] + radius]},
			"zaxis": {"title": "Z", "range": [center[2] - radius, center[2] + radius]},
			"aspectmode": "cube",
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
  
