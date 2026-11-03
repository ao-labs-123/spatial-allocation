import json
from pathlib import Path

import matplotlib.pyplot as plt


def render_graph(graph_data, output_path):
	nodes = graph_data["nodes"]
	positions = {}
	for node in nodes:
		position = node.get("position")
		if not isinstance(position, list) or len(position) != 3:
			raise ValueError(f"Node {node.get('id', '<unknown>')} has no valid 3D position")
		positions[node["id"]] = position

	categories = sorted({node.get("category", "Unknown") for node in nodes})
	colors = plt.get_cmap("tab10")
	category_colors = {category: colors(index % 10) for index, category in enumerate(categories)}

	figure = plt.figure(figsize=(12, 9))
	axis = figure.add_subplot(111, projection="3d")

	for category in categories:
		category_nodes = [node for node in nodes if node.get("category", "Unknown") == category]
		axis.scatter(
			[positions[node["id"]][0] for node in category_nodes],
			[positions[node["id"]][1] for node in category_nodes],
			[positions[node["id"]][2] for node in category_nodes],
			label=category,
			color=[category_colors[category]],
			s=18,
			depthshade=True,
			zorder=2,
		)

	for edge in graph_data["edges"]:
		source = positions.get(edge["source"])
		target = positions.get(edge["target"])
		if source is None or target is None:
			continue
		axis.plot(
			[source[0], target[0]],
			[source[1], target[1]],
			[source[2], target[2]],
			color="#626a73",
			alpha=0.7,
			linewidth=1.4,
			zorder=3,
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
	axis.set_xlim(center[0] - radius, center[0] + radius)
	axis.set_ylim(center[1] - radius, center[1] + radius)
	axis.set_zlim(center[2] - radius, center[2] + radius)
	axis.set_xlabel("X")
	axis.set_ylabel("Y")
	axis.set_zlabel("Z")
	axis.set_title("3D Topological Graph")
	axis.legend(title="Category", loc="upper left", bbox_to_anchor=(1.02, 1))
	figure.tight_layout()
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.show()
	plt.close(figure)


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph_3d.json"
	output_path = project_dir / "visualize" / "topological_graph_3d.png"

	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)

	render_graph(graph_data, output_path)
	print(f"Saved 3D visualization to {output_path}")


if __name__ == "__main__":
	main()
  
