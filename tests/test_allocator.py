import unittest

from src.allocator import allocate_coordinates


class AllocateCoordinatesTests(unittest.TestCase):
	def test_maps_what_similarity_causal_depth_and_temporal_order(self):
		graph_data = {
			"nodes": [
				{"id": "meeting", "attributes": {"what": "long project meeting"}},
				{"id": "after_meeting", "attributes": {"what": "after project meeting"}},
				{"id": "unrelated", "attributes": {"what": "paint a landscape"}},
			],
			"edges": [
				{"source": "meeting", "target": "after_meeting", "morphism_type": "Temporal"},
				{"source": "meeting", "target": "after_meeting", "morphism_type": "Cause"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		meeting_container = nodes[nodes["meeting"]["parent_id"]]
		unrelated_container = nodes[nodes["unrelated"]["parent_id"]]
		self.assertNotEqual(
			meeting_container["semantic_position"][0],
			unrelated_container["semantic_position"][0],
		)
		self.assertEqual(meeting_container["semantic_position"][1], 0.5)
		self.assertEqual(meeting_container["semantic_position"][2], 0.5)
		self.assertEqual(unrelated_container["semantic_position"][1], 0)
		self.assertEqual(unrelated_container["semantic_position"][2], 0)

	def test_explicit_stage_values_create_order(self):
		graph_data = {
			"nodes": [
				{"id": "first", "attributes": {"stage": 1}},
				{"id": "second", "attributes": {"stage": 2}},
			],
			"edges": [],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		first_container = nodes[nodes["first"]["parent_id"]]
		second_container = nodes[nodes["second"]["parent_id"]]
		self.assertLess(first_container["semantic_position"][2], second_container["semantic_position"][2])

	def test_iso_when_values_create_order(self):
		graph_data = {
			"nodes": [
				{"id": "first", "attributes": {"when": "2025-01-01T10:00:00Z"}},
				{"id": "second", "attributes": {"when": "2025-01-02T10:00:00Z"}},
			],
			"edges": [],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		first_container = nodes[nodes["first"]["parent_id"]]
		second_container = nodes[nodes["second"]["parent_id"]]
		self.assertLess(first_container["semantic_position"][2], second_container["semantic_position"][2])

	def test_free_text_when_values_do_not_imply_order(self):
		graph_data = {
			"nodes": [
				{"id": "yesterday", "attributes": {"when": "yesterday"}},
				{"id": "tomorrow", "attributes": {"when": "tomorrow"}},
			],
			"edges": [],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		yesterday = nodes[nodes["yesterday"]["parent_id"]]
		tomorrow = nodes[nodes["tomorrow"]["parent_id"]]
		self.assertEqual(yesterday["semantic_position"][2], tomorrow["semantic_position"][2])

	def test_causal_cycles_are_assigned_the_same_depth(self):
		graph_data = {
			"nodes": [{"id": "a"}, {"id": "b"}],
			"edges": [
				{"source": "a", "target": "b", "morphism_type": "Cause"},
				{"source": "b", "target": "a", "morphism_type": "Hierarchy"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		container = nodes[nodes["a"]["parent_id"]]
		self.assertEqual(nodes["a"]["parent_id"], nodes["b"]["parent_id"])
		self.assertEqual(container["semantic_position"][1], 0)
		self.assertEqual(container["semantic_position"][2], 0)

	def test_temporal_cycles_are_assigned_the_same_stage(self):
		graph_data = {
			"nodes": [{"id": "a"}, {"id": "b"}],
			"edges": [
				{"source": "a", "target": "b", "morphism_type": "Temporal"},
				{"source": "b", "target": "a", "morphism_type": "Temporal"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		container = nodes[nodes["a"]["parent_id"]]
		self.assertEqual(nodes["a"]["parent_id"], nodes["b"]["parent_id"])
		self.assertEqual(container["semantic_position"][1], 0)
		self.assertEqual(container["semantic_position"][2], 0)

	def test_connected_particles_are_nested_in_container_particles(self):
		graph_data = {
			"nodes": [{"id": "a"}, {"id": "b"}, {"id": "isolated"}],
			"edges": [{"source": "a", "target": "b", "morphism_type": "Action"}],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		cluster = nodes["__container__cluster_001"]
		isolated_cluster = nodes["__container__cluster_002"]

		self.assertEqual(cluster["contains"], ["a", "b"])
		self.assertNotIn("parent_id", cluster)
		self.assertEqual(nodes["a"]["parent_id"], cluster["id"])
		self.assertEqual(nodes["b"]["parent_id"], cluster["id"])
		self.assertEqual(nodes["isolated"]["parent_id"], isolated_cluster["id"])
		self.assertEqual(isolated_cluster["contains"], ["isolated"])
		self.assertFalse(any(node.get("container_type") == "conversation" for node in allocated["nodes"]))
		self.assertGreater(cluster["capsule_radius"], 0)
		self.assertGreater(cluster["capsule_half_length"], 0)
		for node_id in cluster["contains"]:
			position = nodes[node_id]["position"]
			center = cluster["position"]
			self.assertLessEqual(abs(position[0] - center[0]), cluster["capsule_half_length"])
			self.assertLessEqual(abs(position[1] - center[1]), cluster["capsule_radius"])
			self.assertLessEqual(abs(position[2] - center[2]), cluster["capsule_radius"])

		allocated_again = allocate_coordinates(allocated)
		self.assertEqual(
			sum(node.get("node_type") == "container" for node in allocated_again["nodes"]),
			2,
		)

	def test_container_capsules_are_packed_without_overlap(self):
		nodes = [{"id": f"node-{index}"} for index in range(8)]
		edges = [
			{"source": f"node-{index}", "target": f"node-{index + 1}", "morphism_type": "Action"}
			for index in range(0, 8, 2)
		]
		allocated = allocate_coordinates({"nodes": nodes, "edges": edges})
		containers = [
			node for node in allocated["nodes"] if node.get("container_type") == "cluster"
		]

		for index, left in enumerate(containers):
			for right in containers[index + 1:]:
				left_position = left["position"]
				right_position = right["position"]
				axial_gap = max(
					0,
					abs(left_position[0] - right_position[0])
					- left["capsule_half_length"]
					- right["capsule_half_length"],
				)
				transverse_distance = (
					(left_position[1] - right_position[1]) ** 2
					+ (left_position[2] - right_position[2]) ** 2
				) ** 0.5
				self.assertGreaterEqual(
					(axial_gap**2 + transverse_distance**2) ** 0.5,
					left["capsule_radius"] + right["capsule_radius"],
				)

	def test_entity_linked_to_mapped_event_becomes_bcc_core(self):
		graph_data = {
			"nodes": [
				{"id": "entity", "label": "Researcher", "category": "Entity"},
				{
					"id": "event",
					"label": "reviewed report",
					"category": "Event",
					"attributes": {"stage2_mapping": "Researcher -> Action -> reviewed report"},
				},
				{"id": "other", "label": "report", "category": "Event"},
			],
			"edges": [
				{"source": "entity", "target": "event", "morphism_type": "Action"},
				{"source": "event", "target": "other", "morphism_type": "Relation"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		container = nodes[nodes["entity"]["parent_id"]]

		self.assertEqual(container["lattice_mode"], "body_centered_cubic")
		self.assertEqual(container["core_indices"][0]["label"], "Researcher")
		self.assertEqual(container["auxiliary_particle_ids"], ["event"])
		self.assertEqual(nodes["entity"]["lattice_role"], "core")
		self.assertEqual(nodes["entity"]["lattice_site"], "body_center")
		self.assertEqual(nodes["event"]["lattice_site"], "vertex")
		self.assertTrue(all(
			value % 1 == 0.5
			for value in nodes["entity"]["lattice_coordinate"]
		))
		self.assertTrue(all(
			value.is_integer()
			for value in nodes["event"]["lattice_coordinate"]
		))
		self.assertEqual(allocated["crystal"]["lattice_mode"], "body_centered_cubic")

	def test_missing_entity_mapping_uses_simple_cubic_sites(self):
		graph_data = {
			"nodes": [
				{"id": "entity", "label": "Unspecified", "category": "Entity"},
				{
					"id": "event",
					"label": "unmapped event",
					"category": "Event",
				},
			],
			"edges": [{"source": "entity", "target": "event", "morphism_type": "Action"}],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		container = nodes[nodes["entity"]["parent_id"]]

		self.assertEqual(container["lattice_mode"], "simple_cubic")
		self.assertEqual(container["core_indices"], [])
		self.assertTrue(all(node["lattice_site"] == "vertex" for node in (nodes["entity"], nodes["event"])))
		self.assertEqual(allocated["crystal"]["preview_repeats"], [3, 3, 3])
		self.assertTrue(allocated["crystal"]["periodic"])
		allocated["crystal"]["preview_repeats"] = [1, 2, 1]
		allocated_again = allocate_coordinates(allocated)
		self.assertEqual(allocated_again["crystal"]["preview_repeats"], [1, 2, 1])
		self.assertEqual(len(allocated_again["nodes"]), len(allocated["nodes"]))

	def test_explicit_lattice_mode_overrides_automatic_classification(self):
		graph_data = {
			"nodes": [
				{"id": "entity", "label": "Researcher", "category": "Entity"},
				{
					"id": "event",
					"category": "Event",
					"attributes": {
						"stage2_mapping": "Researcher -> Action -> reviewed",
						"lattice_mode": "simple_cubic",
					},
				},
			],
			"edges": [{"source": "entity", "target": "event", "morphism_type": "Action"}],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		self.assertEqual(nodes[nodes["entity"]["parent_id"]]["lattice_mode"], "simple_cubic")


if __name__ == "__main__":
	unittest.main()
