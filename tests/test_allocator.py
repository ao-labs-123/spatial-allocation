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
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}

		self.assertLess(
			abs(positions["meeting"][0] - positions["after_meeting"][0]),
			abs(positions["meeting"][0] - positions["unrelated"][0]),
		)
		self.assertEqual(positions["meeting"][1], 0)
		self.assertEqual(positions["after_meeting"][1], 1)
		self.assertEqual(positions["meeting"][2], 0)
		self.assertEqual(positions["after_meeting"][2], 1)
		self.assertEqual(positions["unrelated"][2], 0)

	def test_explicit_stage_values_create_order(self):
		graph_data = {
			"nodes": [
				{"id": "first", "attributes": {"stage": 1}},
				{"id": "second", "attributes": {"stage": 2}},
			],
			"edges": [],
		}

		allocated = allocate_coordinates(graph_data)
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}
		self.assertLess(positions["first"][2], positions["second"][2])

	def test_iso_when_values_create_order(self):
		graph_data = {
			"nodes": [
				{"id": "first", "attributes": {"when": "2025-01-01T10:00:00Z"}},
				{"id": "second", "attributes": {"when": "2025-01-02T10:00:00Z"}},
			],
			"edges": [],
		}

		allocated = allocate_coordinates(graph_data)
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}
		self.assertLess(positions["first"][2], positions["second"][2])

	def test_free_text_when_values_do_not_imply_order(self):
		graph_data = {
			"nodes": [
				{"id": "yesterday", "attributes": {"when": "yesterday"}},
				{"id": "tomorrow", "attributes": {"when": "tomorrow"}},
			],
			"edges": [],
		}

		allocated = allocate_coordinates(graph_data)
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}
		self.assertEqual(positions["yesterday"][2], positions["tomorrow"][2])

	def test_causal_cycles_are_assigned_the_same_depth(self):
		graph_data = {
			"nodes": [{"id": "a"}, {"id": "b"}],
			"edges": [
				{"source": "a", "target": "b", "morphism_type": "Cause"},
				{"source": "b", "target": "a", "morphism_type": "Hierarchy"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}

		self.assertEqual(positions["a"][1], positions["b"][1])

	def test_temporal_cycles_are_assigned_the_same_stage(self):
		graph_data = {
			"nodes": [{"id": "a"}, {"id": "b"}],
			"edges": [
				{"source": "a", "target": "b", "morphism_type": "Temporal"},
				{"source": "b", "target": "a", "morphism_type": "Temporal"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}
		self.assertEqual(positions["a"][2], positions["b"][2])

	def test_connected_particles_are_nested_in_container_particles(self):
		graph_data = {
			"nodes": [{"id": "a"}, {"id": "b"}, {"id": "isolated"}],
			"edges": [{"source": "a", "target": "b", "morphism_type": "Action"}],
		}

		allocated = allocate_coordinates(graph_data)
		nodes = {node["id"]: node for node in allocated["nodes"]}
		root = nodes["__container__conversation"]
		cluster = nodes["__container__cluster_001"]

		self.assertEqual(cluster["contains"], ["a", "b"])
		self.assertEqual(cluster["parent_id"], root["id"])
		self.assertEqual(nodes["a"]["parent_id"], cluster["id"])
		self.assertEqual(nodes["b"]["parent_id"], cluster["id"])
		self.assertEqual(nodes["isolated"]["parent_id"], root["id"])
		self.assertEqual(set(root["contains"]), {cluster["id"], "isolated"})

		allocated_again = allocate_coordinates(allocated)
		self.assertEqual(
			sum(node.get("node_type") == "container" for node in allocated_again["nodes"]),
			2,
		)


if __name__ == "__main__":
	unittest.main()
