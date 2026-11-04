import json
from pathlib import Path

import networkx as nx


UNKNOWN = "Unspecified"
DEPTH_MORPHISMS = {"cause", "hierarchy"}


def _axis_value(node, attribute):
	attributes = node.get("attributes") or {}
	value = attributes.get(attribute)
	if value is None or not str(value).strip():
		return UNKNOWN
	value = str(value).strip()
	return UNKNOWN if value.casefold() == UNKNOWN.casefold() else value


def _normalized_value(value):
	return value.casefold()


def _bucket_coordinates(values):
	unique_values = sorted({_normalized_value(value) for value in values})
	offset = (len(unique_values) - 1) / 2
	return {value: index - offset for index, value in enumerate(unique_values)}


def _causal_depths(nodes, edges):
	graph = nx.DiGraph()
	graph.add_nodes_from(node["id"] for node in nodes)
	for edge in edges:
		if str(edge.get("morphism_type", "")).casefold() not in DEPTH_MORPHISMS:
			continue
		graph.add_edge(edge["source"], edge["target"])

	components = sorted(nx.strongly_connected_components(graph), key=lambda component: tuple(sorted(component)))
	component_for_node = {
		node_id: component_id
		for component_id, component in enumerate(components)
		for node_id in component
	}
	component_graph = nx.DiGraph()
	component_graph.add_nodes_from(range(len(components)))
	for source, target in graph.edges:
		source_component = component_for_node[source]
		target_component = component_for_node[target]
		if source_component != target_component:
			component_graph.add_edge(source_component, target_component)

	depths = {}
	for component in nx.topological_sort(component_graph):
		predecessors = component_graph.predecessors(component)
		depths[component] = max((depths[parent] + 1 for parent in predecessors), default=0)
	return {node_id: depths[component_id] for node_id, component_id in component_for_node.items()}


def allocate_coordinates(graph_data, seed=42):
	nodes = graph_data["nodes"]
	edges = graph_data["edges"]
	node_ids = {node["id"] for node in nodes}

	for edge in edges:
		if edge["source"] not in node_ids or edge["target"] not in node_ids:
			raise ValueError(f"Edge {edge.get('id', '<unknown>')} references an unknown node")

	who_values = {node["id"]: _axis_value(node, "who") for node in nodes}
	when_values = {node["id"]: _axis_value(node, "when") for node in nodes}
	who_coordinates = _bucket_coordinates(who_values.values())
	when_coordinates = _bucket_coordinates(when_values.values())
	depths = _causal_depths(nodes, edges)

	for node in nodes:
		node_id = node["id"]
		node["position"] = [
			float(who_coordinates[_normalized_value(who_values[node_id])]),
			float(depths[node_id]),
			float(when_coordinates[_normalized_value(when_values[node_id])]),
		]

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
