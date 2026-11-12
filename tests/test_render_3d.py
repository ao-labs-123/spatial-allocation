import unittest
from unittest.mock import patch

from visualize.render_3d import (
	MIN_CELLS_PER_AXIS,
	NESTED_PARTICLE_RADIUS,
	SITE_CAPSULE_RADIUS,
	_mixed_lattice_geometry,
	render_lattice,
)


def _graph_data():
	nodes = []
	edges = []
	for index in range(2):
		entity_id = f"entity-{index}"
		event_id = f"mapped-event-{index}"
		nodes.extend(
			[
				{
					"id": entity_id,
					"label": f"Entity label {index}",
					"category": "Entity",
					"attributes": {},
				},
				{
					"id": event_id,
					"label": f"Mapped Event label {index}",
					"category": "Event",
					"attributes": {"stage2_mapping": f"Entity {index} -> Action"},
				},
				{
					"id": f"plain-entity-{index}",
					"label": f"Plain Entity label {index}",
					"category": "Entity",
					"attributes": {},
				},
				{
					"id": f"plain-event-{index}",
					"label": f"Plain Event label {index}",
					"category": "Event",
					"attributes": {},
				},
			]
		)
		edges.extend(
			[
				{
					"id": f"mapped-edge-{index}",
					"source": entity_id,
					"target": event_id,
					"morphism_type": "Action",
				},
				{
					"id": f"plain-edge-{index}",
					"source": f"plain-entity-{index}",
					"target": f"plain-event-{index}",
					"morphism_type": "Relation",
				},
			]
		)
	return {"nodes": nodes, "edges": edges}


class MixedLatticeGeometryTests(unittest.TestCase):
	def test_topology_particles_are_nested_in_vertex_and_bcc_center_sites(self):
		geometry = _mixed_lattice_geometry(_graph_data(), seed=42)
		site_contents = geometry["site_contents"]
		all_nodes = [
			node
			for nodes in site_contents.values()
			for node in nodes
		]

		self.assertEqual(geometry["cells_per_axis"], MIN_CELLS_PER_AXIS)
		self.assertEqual(len(geometry["cell_groups"]), 4)
		self.assertEqual(len(geometry["bcc_cells"]), 2)
		self.assertEqual(len(all_nodes), 8)
		self.assertEqual({node["id"] for node in all_nodes}, {
			"entity-0",
			"mapped-event-0",
			"plain-entity-0",
			"plain-event-0",
			"entity-1",
			"mapped-event-1",
			"plain-entity-1",
			"plain-event-1",
		})
		for nodes in site_contents.values():
			for node in nodes:
				if node["id"].startswith("plain-"):
					self.assertEqual(node["lattice_site"], "vertex")
				elif node["id"].startswith("entity-"):
					self.assertEqual(node["lattice_site"], "body_center")
				else:
					self.assertEqual(node["lattice_site"], "vertex")
		self.assertEqual(
			sum(
				node["category"] == "Entity"
				and node.get("lattice_site") == "body_center"
				for node in all_nodes
			),
			2,
		)

	def test_shared_lattice_vertices_and_bcc_centers_are_the_capsules(self):
		geometry = _mixed_lattice_geometry(_graph_data(), seed=42)
		self.assertEqual(geometry["vertices"].name, "Shared lattice vertices")
		self.assertEqual(geometry["centers"].name, "BCC centers")
		self.assertEqual(geometry["vertices"].type, "mesh3d")
		self.assertEqual(geometry["centers"].type, "mesh3d")
		self.assertEqual(len(geometry["vertices"].x), 4**3 * 86)
		self.assertEqual(len(geometry["centers"].x), 2 * 86)
		self.assertEqual(len(geometry["vertices"].i), 4**3 * 168)
		self.assertEqual(len(geometry["centers"].i), 2 * 168)
		self.assertEqual(
			len(geometry["vertices"].customdata),
			len(geometry["vertices"].i),
		)
		self.assertEqual(
			len(geometry["centers"].customdata),
			len(geometry["centers"].i),
		)
		self.assertTrue(any("nested graph particles" in text for text in geometry["vertices"].text))
		self.assertTrue(any("nested graph particles" in text for text in geometry["centers"].text))
		self.assertAlmostEqual(SITE_CAPSULE_RADIUS, 0.28)
		self.assertAlmostEqual(NESTED_PARTICLE_RADIUS, 0.16)
		self.assertEqual(len(geometry["traces"]), 6)
		self.assertAlmostEqual(geometry["vertices"].x[0], 0)
		self.assertAlmostEqual(geometry["vertices"].y[0], 0)
		self.assertAlmostEqual(
			geometry["vertices"].z[0],
			-SITE_CAPSULE_RADIUS,
		)
		self.assertEqual(geometry["nested_edges"].name, "Nested topology edges")
		self.assertEqual(geometry["nested_particles"].name, "Nested topology particles")
		self.assertTrue(geometry["nested_edges"].x)
		self.assertEqual(
			len(geometry["nested_particles"].x),
			len(geometry["nested_nodes"]),
		)
		self.assertTrue(geometry["nested_particles"].x)

	def test_lattice_bonds_are_connected_and_capsule_data_is_site_scoped(self):
		geometry = _mixed_lattice_geometry(_graph_data(), seed=42)
		adjacency = {}
		for trace in (geometry["simple_cubic_bonds"], geometry["bcc_bonds"]):
			coordinates = list(zip(trace.x, trace.y, trace.z))
			for index in range(0, len(coordinates), 3):
				start, end, separator = coordinates[index:index + 3]
				self.assertIsNone(separator[0])
				adjacency.setdefault(start, set()).add(end)
				adjacency.setdefault(end, set()).add(start)

		self.assertEqual(
			len(geometry["simple_cubic_bonds"].x),
			3 * MIN_CELLS_PER_AXIS * (MIN_CELLS_PER_AXIS + 1) ** 2 * 3,
		)
		self.assertEqual(len(geometry["bcc_bonds"].x), 2 * 8 * 3)
		reached = set()
		pending = [next(iter(adjacency))]
		while pending:
			vertex = pending.pop()
			if vertex in reached:
				continue
			reached.add(vertex)
			pending.extend(adjacency[vertex] - reached)
		self.assertEqual(reached, set(adjacency))
		for site_key, site_data in geometry["site_graph_data"].items():
			self.assertTrue(site_key.startswith(("vertex:", "center:")))
			self.assertTrue(site_data["nodes"])
			center = [
				float(value)
				for value in site_key.split(":", 1)[1].split(",")
			]
			self.assertTrue(all("label" in node for node in site_data["nodes"]))
			self.assertTrue(all(
				len(node["position"]) == 3
				for node in site_data["nodes"]
			))
			self.assertTrue(all(
				sum((node["position"][axis] - center[axis]) ** 2 for axis in range(3))
				<= NESTED_PARTICLE_RADIUS**2 + 1e-12
				for node in site_data["nodes"]
			))
			self.assertTrue(all(
				edge["source"] in {node["id"] for node in site_data["nodes"]}
				and edge["target"] in {node["id"] for node in site_data["nodes"]}
				and len(edge["source_position"]) == 3
				and len(edge["target_position"]) == 3
				for edge in site_data["edges"]
			))

	def test_renderer_shows_three_dimensional_particles_without_clicking(self):
		with patch(
			"visualize.render_3d.go.Figure.write_html",
			autospec=True,
		) as write_html:
			render_lattice(_graph_data(), "unused.html", seed=42)

		figure = write_html.call_args.args[0]
		self.assertEqual(
			[trace.name for trace in figure.data],
			[
				"Simple cubic bonds",
				"BCC body-center bonds",
				"Shared lattice vertices",
				"BCC centers",
				"Nested topology edges",
				"Nested topology particles",
			],
		)
		self.assertIn("3 x 3 x 3", figure.layout.title.text)
		self.assertIn("visible nested topology", figure.layout.title.text)
		site_data = _mixed_lattice_geometry(_graph_data(), seed=42)["site_graph_data"]
		self.assertEqual(
			len(figure.data[5].x),
			sum(len(data["nodes"]) for data in site_data.values()),
		)
		self.assertEqual(
			len(figure.data[4].x),
			sum(3 * len(data["edges"]) for data in site_data.values()),
		)
		self.assertTrue(figure.data[4].x)
		self.assertTrue(figure.data[5].x)
		labels = {
			label
			for data in _mixed_lattice_geometry(_graph_data(), seed=42)["site_graph_data"].values()
			for label in [node["label"] for node in data["nodes"]]
		}
		self.assertEqual(
			labels,
			{
				"Entity label 0",
				"Mapped Event label 0",
				"Plain Entity label 0",
				"Plain Event label 0",
				"Entity label 1",
				"Mapped Event label 1",
				"Plain Entity label 1",
				"Plain Event label 1",
			},
		)
		self.assertNotIn("post_script", write_html.call_args.kwargs)


if __name__ == "__main__":
	unittest.main()
