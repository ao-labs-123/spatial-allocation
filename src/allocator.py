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
CRYSTAL_PREVIEW_REPEATS = (3, 3, 3)


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


def _mapping_fields(node):
	attributes = node.get("attributes") or {}
	return {
		field: value.strip()
		for field, value in attributes.items()
		if field.endswith("_mapping")
		and isinstance(value, str)
		and value.strip()
		and value.strip().casefold() != UNKNOWN.casefold()
	}


def _entity_index_label(node):
	label = str(node.get("label") or "").strip()
	return None if not label or label.casefold() == UNKNOWN.casefold() else label


def _classify_lattice(member_ids, node_by_id, graph):
	core_indices = []
	auxiliary_particle_ids = set()
	cores_by_event = {}

	for node_id in member_ids:
		node = node_by_id[node_id]
		if str(node.get("category", "")).casefold() != "entity":
			continue
		label = _entity_index_label(node)
		if label is None:
			continue

		mapped_events = sorted(
			neighbor_id
			for neighbor_id in graph.neighbors(node_id)
			if str(node_by_id[neighbor_id].get("category", "")).casefold() == "event"
			and _mapping_fields(node_by_id[neighbor_id])
		)
		if not mapped_events:
			continue

		core_indices.append({
			"node_id": node_id,
			"label": label,
			"mapped_event_ids": mapped_events,
		})
		node["lattice_role"] = "core"
		node["lattice_index"] = label
		for event_id in mapped_events:
			auxiliary_particle_ids.add(event_id)
			cores_by_event.setdefault(event_id, []).append({
				"node_id": node_id,
				"label": label,
			})

	for event_id, cores in cores_by_event.items():
		node_by_id[event_id]["lattice_role"] = "vertex"
		node_by_id[event_id]["lattice_cores"] = sorted(
			cores,
			key=lambda core: (core["label"].casefold(), core["node_id"]),
		)

	core_indices.sort(key=lambda core: (core["label"].casefold(), core["node_id"]))
	return {
		"lattice_mode": "body_centered_cubic" if core_indices else "simple_cubic",
		"core_indices": core_indices,
		"auxiliary_particle_ids": sorted(auxiliary_particle_ids),
	}


def _grid_dimensions(count):
	columns = math.ceil(count ** (1 / 3))
	rows = math.ceil(math.sqrt(count / columns))
	layers = math.ceil(count / (columns * rows))
	return columns, rows, layers


def _lattice_sites(dimensions, body_centered):
	columns, rows, layers = dimensions
	corners = set()
	centers = []
	for x_index in range(columns):
		for y_index in range(rows):
			for z_index in range(layers):
				origin = (x_index, y_index, z_index)
				corners.update(
					(
						origin[0] + x_offset,
						origin[1] + y_offset,
						origin[2] + z_offset,
					)
					for x_offset in (0, 1)
					for y_offset in (0, 1)
					for z_offset in (0, 1)
				)
				centers.append(tuple(value + 0.5 for value in origin))

	if body_centered:
		return corners, centers
	return corners, []


def _simple_cubic_sites(count):
	dimensions = _grid_dimensions(count)
	columns, rows, layers = dimensions
	center = ((columns - 1) / 2, (rows - 1) / 2, (layers - 1) / 2)
	sites = [
		(float(x_index), float(y_index), float(z_index))
		for x_index in range(columns)
		for y_index in range(rows)
		for z_index in range(layers)
	]
	return sorted(sites, key=lambda point: (_distance_squared(point, center), point))


def _distance_squared(left, right):
	return sum((left[axis] - right[axis]) ** 2 for axis in range(3))


def _attach_source_examples(nodes, edges, source_logs):
	if source_logs is None:
		return
	if not isinstance(source_logs, list):
		raise ValueError("Source logs must be a list of analysis records")

	examples = {}
	mapping_sources = {}
	for index, record in enumerate(source_logs, start=1):
		if not isinstance(record, dict) or not isinstance(record.get("input"), str):
			raise ValueError(f"Source log entry {index} must contain an input string")
		example_id = f"example-{index:03d}"
		examples[example_id] = record["input"]
		for stage in ("stage2", "stage3"):
			mapping = (record.get(stage) or {}).get("mapping")
			if isinstance(mapping, str) and mapping.strip():
				mapping_sources.setdefault((stage, mapping.strip()), set()).add(example_id)

	node_sources = {}
	node_by_id = {node["id"]: node for node in nodes}
	for node in nodes:
		attributes = node.get("attributes") or {}
		candidate_sets = []
		for field, stage in (("stage2_mapping", "stage2"), ("stage3_mapping", "stage3")):
			mapping = attributes.get(field)
			if isinstance(mapping, str) and mapping.strip():
				candidate_sets.append(mapping_sources.get((stage, mapping.strip()), set()))
		matches = set.intersection(*candidate_sets) if candidate_sets else set()
		if len(matches) == 1:
			node_sources[node["id"]] = next(iter(matches))

	graph = nx.Graph()
	graph.add_nodes_from(node["id"] for node in nodes)
	graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
	for component in nx.connected_components(graph):
		matched_sources = {node_sources[node_id] for node_id in component if node_id in node_sources}
		if len(matched_sources) == 1:
			source_id = next(iter(matched_sources))
		elif not matched_sources:
			labels = {
				str(node_by_id[node_id].get("label", "")).strip().casefold()
				for node_id in component
			}
			matches = []
			for example_id, record in zip(examples, source_logs):
				structure = (record.get("stage2") or {}).get("structure") or {}
				relation_labels = {
					str(structure.get(field, "")).strip().casefold()
					for field in ("cause", "effect")
					if structure.get(field)
				}
				if len(labels) > 1 and labels == relation_labels:
					matches.append(example_id)
			if len(matches) != 1:
				continue
			source_id = matches[0]
		else:
			continue

		for node_id in component:
			node_sources[node_id] = source_id

	for node in nodes:
		source_id = node_sources.get(node["id"])
		if source_id is not None:
			node["source_example_id"] = source_id
			node["source_text"] = examples[source_id]


def _crystal_supercell_metadata(nodes):
	contents = [
		node for node in nodes
		if node.get("node_type") != "container"
		or node.get("container_type") == "cluster"
	]
	bounds = [[float("inf"), float("-inf")] for _ in range(3)]
	for node in contents:
		position = node.get("position")
		if not isinstance(position, list) or len(position) != 3:
			continue
		if node.get("container_type") == "cluster":
			radius = float(node["capsule_radius"])
			extents = [float(node["capsule_half_length"]) + radius, radius, radius]
		else:
			extents = [0.0, 0.0, 0.0]
		for axis in range(3):
			bounds[axis][0] = min(bounds[axis][0], float(position[axis]) - extents[axis])
			bounds[axis][1] = max(bounds[axis][1], float(position[axis]) + extents[axis])

	if not contents or any(not math.isfinite(bound) for axis_bounds in bounds for bound in axis_bounds):
		bounds = [[-1.0, 1.0] for _ in range(3)]

	margin = 0.75
	center = [(lower + upper) / 2 for lower, upper in bounds]
	size = [upper - lower + 2 * margin for lower, upper in bounds]
	source_examples = {
		node["source_example_id"]: node["source_text"]
		for node in nodes
		if node.get("source_example_id") and node.get("source_text")
	}
	return {
		"basis_type": "dataset_supercell",
		"preview_repeats": list(CRYSTAL_PREVIEW_REPEATS),
		"unit_cell": {
			"center": center,
			"size": size,
			"bounds": [
				[center[axis] - size[axis] / 2, center[axis] + size[axis] / 2]
				for axis in range(3)
			],
			"source_examples": [
				{"id": example_id, "input": source_examples[example_id]}
				for example_id in sorted(source_examples)
			],
		},
	}


def _lattice_particle_positions(member_ids, node_by_id, container):
	body_centered = container["lattice_mode"] == "body_centered_cubic"
	core_indices = container["core_indices"]
	if body_centered:
		cell_count = max(len(core_indices), math.ceil(len(member_ids) / 2), 1)
		dimensions = _grid_dimensions(cell_count)
		corners, centers = _lattice_sites(dimensions, True)
		grid_center = tuple((dimension - 1) / 2 for dimension in dimensions)
		ordered_corners = sorted(
			corners,
			key=lambda point: (_distance_squared(point, grid_center), point),
		)
		ordered_centers = sorted(
			centers,
			key=lambda point: (_distance_squared(point, grid_center), point),
		)
	else:
		ordered_corners = _simple_cubic_sites(len(member_ids))
		ordered_centers = []
	assignments = {}
	site_kinds = {}

	if body_centered:
		assigned_core_positions = {}
		for core, position in zip(core_indices, ordered_centers):
			node_id = core["node_id"]
			assignments[node_id] = position
			site_kinds[node_id] = "body_center"
			assigned_core_positions[node_id] = position

		event_cores = {}
		for core in core_indices:
			for event_id in core["mapped_event_ids"]:
				event_cores.setdefault(event_id, []).append(core["node_id"])
		available_corners = set(corners)
		for event_id in sorted(event_cores):
			core_ids = event_cores[event_id]
			core_position = assigned_core_positions[core_ids[0]]
			position = min(
				available_corners,
				key=lambda candidate: (_distance_squared(candidate, core_position), candidate),
			)
			assignments[event_id] = position
			site_kinds[event_id] = "vertex"
			available_corners.remove(position)

		remaining_ids = [node_id for node_id in member_ids if node_id not in assignments]
		remaining_sites = [
			(point, "vertex") for point in ordered_corners if point in available_corners
		]
		remaining_sites.extend(
			(point, "body_center")
			for point in ordered_centers
			if point not in set(assigned_core_positions.values())
		)
	else:
		remaining_ids = list(member_ids)
		remaining_sites = [(point, "vertex") for point in ordered_corners]

	for node_id, (position, site_kind) in zip(remaining_ids, remaining_sites):
		assignments[node_id] = position
		site_kinds[node_id] = site_kind

	axis_centers = [
		(
			min(position[axis] for position in assignments.values())
			+ max(position[axis] for position in assignments.values())
		) / 2
		if assignments else 0.0
		for axis in range(3)
	]
	axis_extents = [
		max((abs(position[axis] - axis_centers[axis]) for position in assignments.values()), default=0.0)
		for axis in range(3)
	]
	center = container["position"]
	target_extents = [
		container["capsule_half_length"] * 0.72,
		container["capsule_radius"] * 0.62,
		container["capsule_radius"] * 0.62,
	]
	for node_id, coordinate in assignments.items():
		node = node_by_id[node_id]
		node["lattice_coordinate"] = [float(value) for value in coordinate]
		node["lattice_site"] = site_kinds[node_id]
		node["position"] = [
			float(center[axis])
			+ (
				(coordinate[axis] - axis_centers[axis]) / axis_extents[axis] * target_extents[axis]
				if axis_extents[axis] > 1e-12
				else 0.0
			)
			for axis in range(3)
		]


def _add_container_particles(nodes, edges, seed=42):
	if not nodes:
		return nodes

	graph = nx.Graph()
	graph.add_nodes_from(node["id"] for node in nodes)
	graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
	components = sorted(
		nx.connected_components(graph),
		key=lambda component: tuple(sorted(component)),
	)
	root_id = "__container__conversation"
	if any(node["id"] == root_id for node in nodes):
		raise ValueError(f"Node ID {root_id!r} is reserved for the conversation container")

	top_level_members = []
	container_nodes = []
	node_by_id = {node["id"]: node for node in nodes}
	component_layouts = []
	for cluster_index, component in enumerate(components, start=1):
		member_ids = sorted(component)
		container_id = f"__container__cluster_{cluster_index:03d}"
		if any(node["id"] == container_id for node in nodes):
			raise ValueError(f"Node ID {container_id!r} is reserved for a cluster container")
		member_positions = [node_by_id[node_id]["position"] for node_id in member_ids]
		center = [
			sum(position[axis] for position in member_positions) / len(member_positions)
			for axis in range(3)
		]
		radius = 0.75 + 0.16 * math.sqrt(len(member_ids))
		half_length = max(0.3, 0.28 * (len(member_ids) - 1))
		lattice = _classify_lattice(member_ids, node_by_id, graph)
		source_example_ids = sorted({
			node_by_id[node_id]["source_example_id"]
			for node_id in member_ids
			if node_by_id[node_id].get("source_example_id")
		})
		for node_id in member_ids:
			node_by_id[node_id]["topology_cluster_id"] = container_id

		container = {
			"id": container_id,
			"label": f"Local cluster {cluster_index}",
			"category": "Container",
			"node_type": "container",
			"container_type": "cluster",
			"resolution_state": "Derived",
			"contains": member_ids,
			"parent_id": root_id,
			"capsule_radius": radius,
			"capsule_half_length": half_length,
			"semantic_position": center,
			"source_example_ids": source_example_ids,
			**lattice,
		}
		container_nodes.append(container)
		component_layouts.append((container, member_ids))

	if component_layouts:
		max_extent = max(
			container["capsule_radius"] + container["capsule_half_length"]
			for container, _ in component_layouts
		)
		spacing = 2 * max_extent + 0.5
		columns = math.ceil(len(component_layouts) ** (1 / 3))
		rows = math.ceil(math.sqrt(len(component_layouts) / columns))
		layers = math.ceil(len(component_layouts) / (columns * rows))
		for index, (container, member_ids) in enumerate(
			sorted(
				component_layouts,
				key=lambda item: (
					*item[0]["semantic_position"],
					item[0]["id"],
				),
			)
		):
			grid_x = index % columns
			grid_y = (index // columns) % rows
			grid_z = index // (columns * rows)
			container["position"] = [
				(grid_x - (columns - 1) / 2) * spacing,
				(grid_y - (rows - 1) / 2) * spacing,
				(grid_z - (layers - 1) / 2) * spacing,
			]
			center = container["position"]
			_lattice_particle_positions(member_ids, node_by_id, container)
			for node_id in member_ids:
				node = node_by_id[node_id]
				node["parent_id"] = container["id"]
			top_level_members.append(container["id"])

	root = {
		"id": root_id,
		"label": "Conversation",
		"category": "Container",
		"node_type": "container",
		"container_type": "conversation",
		"resolution_state": "Derived",
		"contains": sorted(top_level_members),
		"semantic_position": [
			sum(container["semantic_position"][axis] for container, _ in component_layouts)
			/ len(component_layouts)
			for axis in range(3)
		] if component_layouts else [0.0, 0.0, 0.0],
	}
	root["position"] = (
		[
			0.0,
			max(container["position"][1] for container in container_nodes) + spacing,
			0.0,
		]
		if container_nodes
		else [0.0, 0.0, 0.0]
	)

	return nodes + container_nodes + [root]


def allocate_coordinates(graph_data, seed=42, source_logs=None):
	nodes = [node for node in graph_data["nodes"] if node.get("node_type") != "container"]
	for node in nodes:
		node.pop("parent_id", None)
		node.pop("topology_cluster_id", None)
		for field in ("lattice_role", "lattice_index", "lattice_cores"):
			node.pop(field, None)
	graph_data["nodes"] = nodes
	edges = graph_data["edges"]
	node_ids = {node["id"] for node in nodes}

	for edge in edges:
		if edge["source"] not in node_ids or edge["target"] not in node_ids:
			raise ValueError(f"Edge {edge.get('id', '<unknown>')} references an unknown node")

	_attach_source_examples(nodes, edges, source_logs)
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

	graph_data["nodes"] = _add_container_particles(nodes, edges, seed=seed)
	graph_data["crystal"] = _crystal_supercell_metadata(graph_data["nodes"])
	return graph_data


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph.json"
	source_log_path = project_dir / "data" / "source" / "log.json"
	output_path = project_dir / "data" / "topological_graph_3d.json"

	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)
	with source_log_path.open(encoding="utf-8") as source_log_file:
		source_logs = json.load(source_log_file)

	graph_data = allocate_coordinates(graph_data, source_logs=source_logs)

	with output_path.open("w", encoding="utf-8") as output_file:
		json.dump(graph_data, output_file, ensure_ascii=False, indent=2)
		output_file.write("\n")

	print(f"Saved 3D coordinates for {len(graph_data['nodes'])} nodes to {output_path}")


if __name__ == "__main__":
	main()
