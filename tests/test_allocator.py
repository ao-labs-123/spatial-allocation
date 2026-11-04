import unittest

from src.allocator import allocate_coordinates


class AllocateCoordinatesTests(unittest.TestCase):
	def test_maps_who_when_and_causal_depth_to_axes(self):
		graph_data = {
			"nodes": [
				{"id": "cause", "attributes": {"who": "I", "when": "yesterday"}},
				{"id": "effect", "attributes": {"who": "I", "when": "yesterday"}},
				{"id": "other", "attributes": {"who": "You", "when": "Unspecified"}},
			],
			"edges": [
				{"source": "cause", "target": "effect", "morphism_type": "Cause"},
				{"source": "effect", "target": "other", "morphism_type": "Temporal"},
			],
		}

		allocated = allocate_coordinates(graph_data)
		positions = {node["id"]: node["position"] for node in allocated["nodes"]}

		self.assertEqual(positions["cause"][0], positions["effect"][0])
		self.assertNotEqual(positions["cause"][0], positions["other"][0])
		self.assertEqual(positions["cause"][1], 0)
		self.assertEqual(positions["effect"][1], 1)
		self.assertEqual(positions["other"][1], 0)
		self.assertEqual(positions["cause"][2], positions["effect"][2])
		self.assertNotEqual(positions["cause"][2], positions["other"][2])

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


if __name__ == "__main__":
	unittest.main()
