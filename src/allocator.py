import json
from pathlib import Path

import networkx as nx


def allocate_coordinates(graph_data, seed=42):
	nodes = graph_data["nodes"]
	edges = graph_data["edges"]
	node_ids = {node["id"] for node in nodes}

	graph = nx.Graph()
	graph.add_nodes_from(node_ids)

	for edge in edges:
		source = edge["source"]
		target = edge["target"]
		if source not in node_ids or target not in node_ids:
			raise ValueError(f"Edge {edge.get('id', '<unknown>')} references an unknown node")
		graph.add_edge(source, target)

	positions = nx.spring_layout(graph, dim=3, seed=seed)
	for node in nodes:
		node["position"] = [float(value) for value in positions[node["id"]]]

	return graph_data


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph.json"
	output_path = project_dir / "data" / "topological_graph_3d.json"

	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)

	graph_data = allocate_coordinates(graph_data)

	with output_path.open("w", encoding="utf-8") as output_file:
		json.dump(graph_data, output_file, ensure_ascii=False, indent=2)
		output_file.write("\n")

	print(f"Saved 3D coordinates for {len(graph_data['nodes'])} nodes to {output_path}")


if __name__ == "__main__":
	main()
  
