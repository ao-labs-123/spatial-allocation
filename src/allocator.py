import json
import math
from datetime import datetime, timezone
from pathlib import Path
import re

import networkx as nx
import numpy as np


UNKNOWN = "Unspecified"
DEPTH_MORPHISMS = {"cause", "hierarchy"}
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’-][^\W_]+)*", re.UNICODE)
STAGE_FIELDS = ("stage", "stage_index", "process_stage")


def _axis_value(node, attribute):
	attributes = node.get("attributes") or {}
	value = attributes.get(attribute)
	if value is None or not str(value).strip():
		return UNKNOWN
	value = str(value).strip()
	return UNKNOWN if value.casefold() == UNKNOWN.casefold() else value


def _normalized_value(value):
	return value.casefold()


def _what_coordinates(nodes):
	values = {}
	for node in nodes:
		value = _axis_value(node, "what")
		values[node["id"]] = value

	known_values = sorted({_normalized_value(value) for value in values.values() if value != UNKNOWN})
	if not known_values:
		return {node_id: 0.0 for node_id in values}

	tokens_by_value = {
		value: set(TOKEN_PATTERN.findall(value.casefold()))
		for value in known_values
	}
	document_frequency = {}
	for tokens in tokens_by_value.values():
		for token in tokens:
			document_frequency[token] = document_frequency.get(token, 0) + 1

	weights_by_value = {}
	for value, tokens in tokens_by_value.items():
		weights = {
			token: (1 + math.log(len(known_values) / (1 + document_frequency[token])))
			for token in tokens
		}
		norm = math.sqrt(sum(weight * weight for weight in weights.values()))
		weights_by_value[value] = (
			{token: weight / norm for token, weight in weights.items()} if norm else {}
		)

	count = len(known_values)
	distances = np.zeros((count, count), dtype=float)
	for left in range(count):
		for right in range(left + 1, count):
			left_weights = weights_by_value[known_values[left]]
			right_weights = weights_by_value[known_values[right]]
			similarity = sum(
				weight * right_weights.get(token, 0.0)
				for token, weight in left_weights.items()
			)
			distance = math.sqrt(max(0.0, 1.0 - similarity))
			distances[left, right] = distance
			distances[right, left] = distance

	if count == 1 or not np.any(distances):
		known_coordinates = np.zeros(count)
	else:
		centering = np.eye(count) - np.ones((count, count)) / count
		gram = -0.5 * centering @ (distances * distances) @ centering
		eigenvalues, eigenvectors = np.linalg.eigh(gram)
		leading_index = int(np.argmax(eigenvalues))
		leading_value = max(0.0, float(eigenvalues[leading_index]))
		known_coordinates = eigenvectors[:, leading_index] * math.sqrt(leading_value)
		orientation_index = next(
			(index for index, coordinate in enumerate(known_coordinates) if abs(coordinate) > 1e-12),
			None,
		)
		if orientation_index is not None and known_coordinates[orientation_index] < 0:
			known_coordinates *= -1

	value_coordinates = {
		value: float(known_coordinates[index])
		for index, value in enumerate(known_values)
	}
	unspecified_coordinate = float(np.median(known_coordinates))
	return {
		node_id: (
			unspecified_coordinate
			if value == UNKNOWN
			else value_coordinates[_normalized_value(value)]
		)
		for node_id, value in values.items()
	}


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


def _explicit_stage(node):
	attributes = node.get("attributes") or {}
	for field in STAGE_FIELDS:
		value = attributes.get(field)
		if value is not None and str(value).strip():
			try:
				return ("number", float(value))
			except (TypeError, ValueError) as error:
				raise ValueError(f"Node {node['id']} has a non-numeric {field}: {value!r}") from error

	when = _axis_value(node, "when")
	if when != UNKNOWN:
		try:
			timestamp = datetime.fromisoformat(when.replace("Z", "+00:00"))
			if timestamp.tzinfo is None:
				timestamp = timestamp.replace(tzinfo=timezone.utc)
			return ("timestamp", timestamp.timestamp())
		except (TypeError, ValueError):
			pass
	return None


def _temporal_stages(nodes, edges):
	stage_values = {node["id"]: _explicit_stage(node) for node in nodes}
	stage_kinds = {value[0] for value in stage_values.values() if value is not None}
	if len(stage_kinds) > 1:
		raise ValueError("Explicit stages and timestamps cannot be mixed in one graph")

	graph = nx.DiGraph()
	graph.add_nodes_from(node["id"] for node in nodes)
	for edge in edges:
		if str(edge.get("morphism_type", "")).casefold() == "temporal":
			graph.add_edge(edge["source"], edge["target"])

	ordered_stages = sorted({value[1] for value in stage_values.values() if value is not None})
	stage_nodes = {
		value: [node_id for node_id, stage in stage_values.items() if stage is not None and stage[1] == value]
		for value in ordered_stages
	}
	for earlier_index, earlier in enumerate(ordered_stages):
		for later in ordered_stages[earlier_index + 1:]:
			for source in stage_nodes[earlier]:
				for target in stage_nodes[later]:
					graph.add_edge(source, target)

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
	return {node_id: float(depths[component_id]) for node_id, component_id in component_for_node.items()}


def allocate_coordinates(graph_data, seed=42):
	nodes = graph_data["nodes"]
	edges = graph_data["edges"]
	node_ids = {node["id"] for node in nodes}

	for edge in edges:
		if edge["source"] not in node_ids or edge["target"] not in node_ids:
			raise ValueError(f"Edge {edge.get('id', '<unknown>')} references an unknown node")

	depths = _causal_depths(nodes, edges)
	what_coordinates = _what_coordinates(nodes)
	temporal_stages = _temporal_stages(nodes, edges)

	for node in nodes:
		node_id = node["id"]
		node["position"] = [
			what_coordinates[node_id],
			float(depths[node_id]),
			temporal_stages[node_id],
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
