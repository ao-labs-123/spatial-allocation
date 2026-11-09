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


def _classify_lattice(member_ids, node_by_id, graph):
	core_indices = []
	auxiliary_particle_ids = set()
	cores_by_event = {}

	for node_id in member_ids:
		node = node_by_id[node_id]
		label = str(node.get("label") or "").strip()
		if (
			str(node.get("category", "")).casefold() != "entity"
			or not label
			or label.casefold() == UNKNOWN.casefold()
		):
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
	explicit_modes = {
		str((node_by_id[node_id].get("attributes") or {}).get("lattice_mode", "")).strip().casefold()
		for node_id in member_ids
		if (node_by_id[node_id].get("attributes") or {}).get("lattice_mode") is not None
	}
	mode_aliases = {
		"sc": "simple_cubic",
		"simple_cubic": "simple_cubic",
		"bcc": "body_centered_cubic",
		"body_centered_cubic": "body_centered_cubic",
	}
	if len(explicit_modes) > 1:
		raise ValueError(f"Conflicting lattice_mode values in component: {sorted(explicit_modes)}")
	if explicit_modes:
		explicit_mode = next(iter(explicit_modes))
		if explicit_mode not in mode_aliases:
			raise ValueError(f"Unsupported lattice_mode: {explicit_mode!r}")
		lattice_mode = mode_aliases[explicit_mode]
	else:
		lattice_mode = "body_centered_cubic" if core_indices else "simple_cubic"

	return {
		"lattice_mode": lattice_mode,
		"core_indices": core_indices,
		"auxiliary_particle_ids": sorted(auxiliary_particle_ids),
	}


def _grid_dimensions(count):
	columns = math.ceil(count ** (1 / 3))
	rows = math.ceil(math.sqrt(count / columns))
	layers = math.ceil(count / (columns * rows))
	return columns, rows, layers


def _lattice_particle_positions(member_ids, node_by_id, container):
	body_centered = container["lattice_mode"] == "body_centered_cubic"
	core_indices = container["core_indices"]
	if body_centered:
		cell_count = max(len(core_indices), math.ceil(len(member_ids) / 2), 1)
		columns, rows, layers = _grid_dimensions(cell_count)
		corners = {
			(float(x), float(y), float(z))
			for x in range(columns + 1)
			for y in range(rows + 1)
			for z in range(layers + 1)
		}
		centers = [
			(x + 0.5, y + 0.5, z + 0.5)
			for x in range(columns)
			for y in range(rows)
			for z in range(layers)
		]
		grid_center = (columns / 2, rows / 2, layers / 2)
		ordered_corners = sorted(
			corners,
			key=lambda point: (
				sum((point[axis] - grid_center[axis]) ** 2 for axis in range(3)),
				point,
			),
		)
		ordered_centers = sorted(
			centers,
			key=lambda point: (
				sum((point[axis] - grid_center[axis]) ** 2 for axis in range(3)),
				point,
			),
		)
	else:
		columns, rows, layers = _grid_dimensions(len(member_ids))
		ordered_corners = [
			(float(x), float(y), float(z))
			for x in range(columns)
			for y in range(rows)
			for z in range(layers)
		]
		ordered_centers = []

	assignments = {}
	site_kinds = {}
	remaining_ids = list(member_ids)
	if body_centered:
		for core, position in zip(core_indices, ordered_centers):
			node_id = core["node_id"]
			assignments[node_id] = position
			site_kinds[node_id] = "body_center"
			remaining_ids.remove(node_id)

		event_cores = {}
		for core in core_indices:
			for event_id in core["mapped_event_ids"]:
				event_cores.setdefault(event_id, []).append(core["node_id"])
		available_corners = set(corners)
		for event_id in sorted(event_cores):
			if event_id not in remaining_ids or not available_corners:
				continue
			core_position = assignments[event_cores[event_id][0]]
			position = min(
				available_corners,
				key=lambda candidate: (
					sum((candidate[axis] - core_position[axis]) ** 2 for axis in range(3)),
					candidate,
				),
			)
			assignments[event_id] = position
			site_kinds[event_id] = "vertex"
			remaining_ids.remove(event_id)
			available_corners.remove(position)

		remaining_sites = [
			(point, "vertex") for point in ordered_corners if point in available_corners
		]
		remaining_sites.extend(
			(point, "body_center")
			for point in ordered_centers
			if point not in set(assignments.values())
		)
	else:
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


def _crystal_metadata(containers, preview_repeats):
	if (
		not isinstance(preview_repeats, list)
		or len(preview_repeats) != 3
		or any(not isinstance(count, int) or isinstance(count, bool) or count < 1 for count in preview_repeats)
	):
		raise ValueError("Crystal preview_repeats must contain three positive integers")
	bounds = [[float("inf"), float("-inf")] for _ in range(3)]
	for container in containers:
		position = container["position"]
		radius = container["capsule_radius"]
		extents = [container["capsule_half_length"] + radius, radius, radius]
		for axis in range(3):
			bounds[axis][0] = min(bounds[axis][0], position[axis] - extents[axis])
			bounds[axis][1] = max(bounds[axis][1], position[axis] + extents[axis])

	if not containers:
		bounds = [[-1.0, 1.0] for _ in range(3)]
	center = [(lower + upper) / 2 for lower, upper in bounds]
	cubic_size = max(upper - lower for lower, upper in bounds) + 1.0
	return {
		"basis_type": "dataset_supercell",
		"periodic": True,
		"lattice_mode": (
			"body_centered_cubic"
			if any(container["lattice_mode"] == "body_centered_cubic" for container in containers)
			else "simple_cubic"
		),
		"preview_repeats": list(preview_repeats),
		"unit_cell": {
			"center": center,
			"size": [cubic_size, cubic_size, cubic_size],
		},
	}


def _add_container_particles(nodes, edges, seed=42, preview_repeats=(2, 2, 2)):
	if not nodes:
		return nodes

	graph = nx.Graph()
	graph.add_nodes_from(node["id"] for node in nodes)
	graph.add_edges_from((edge["source"], edge["target"]) for edge in edges)
	components = sorted(
		nx.connected_components(graph),
		key=lambda component: tuple(sorted(component)),
	)
	container_nodes = []
	node_by_id = {node["id"]: node for node in nodes}
	component_layouts = []
	for node in nodes:
		for field in ("lattice_role", "lattice_index", "lattice_cores", "lattice_coordinate", "lattice_site"):
			node.pop(field, None)
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

		container = {
			"id": container_id,
			"label": f"Local cluster {cluster_index}",
			"category": "Container",
			"node_type": "container",
			"container_type": "cluster",
			"resolution_state": "Derived",
			"contains": member_ids,
			"capsule_radius": radius,
			"capsule_half_length": half_length,
			"semantic_position": center,
			**_classify_lattice(member_ids, node_by_id, graph),
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
			for node_id in member_ids:
				node = node_by_id[node_id]
				node["parent_id"] = container["id"]
			_lattice_particle_positions(member_ids, node_by_id, container)
	return nodes + container_nodes


def allocate_coordinates(graph_data, seed=42):
	nodes = [node for node in graph_data["nodes"] if node.get("node_type") != "container"]
	for node in nodes:
		node.pop("parent_id", None)
	graph_data["nodes"] = nodes
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

	crystal_config = graph_data.get("crystal") or {}
	if not isinstance(crystal_config, dict):
		raise ValueError("Graph crystal metadata must be an object")
	preview_repeats = crystal_config.get("preview_repeats", [2, 2, 2])
	graph_data["nodes"] = _add_container_particles(
		nodes,
		edges,
		seed=seed,
		preview_repeats=preview_repeats,
	)
	cluster_containers = [
		node for node in graph_data["nodes"]
		if node.get("container_type") == "cluster"
	]
	if cluster_containers:
		graph_data["crystal"] = _crystal_metadata(cluster_containers, preview_repeats)
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
