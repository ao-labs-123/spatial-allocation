import unittest
from unittest.mock import patch

from visualize.render_3d import (
	CELLS_PER_AXIS,
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
	def test_graph_components_are_assigned_to_random_sc_or_bcc_cells(self):
		graph = _graph_data()
		first = _mixed_lattice_geometry(graph, seed=42)
		second = _mixed_lattice_geometry(graph, seed=42)

		self.assertEqual(CELLS_PER_AXIS, 3)
		self.assertEqual(first["bcc_cells"], second["bcc_cells"])
		self.assertEqual(len(first["bcc_cells"]), 2)
		self.assertEqual(len(first["cell_groups"]), 4)
		self.assertEqual(
			sum(len(groups) for groups in first["cell_groups"].values()),
			4,
		)
		self.assertEqual(len(first["bcc_bonds"].x), 2 * 8 * 3)
		self.assertEqual(len(first["simple_cubic_bonds"].x), 3 * 3 * 4**2 * 3)
		self.assertEqual(len(first["vertices"].x), 4**3)

	def test_lattice_and_bcc_bonds_are_connected(self):
		geometry = _mixed_lattice_geometry(_graph_data(), seed=42)
		adjacency = {}
		for trace in (geometry["simple_cubic_bonds"], geometry["bcc_bonds"]):
			coordinates = list(zip(trace.x, trace.y, trace.z))
			for index in range(0, len(coordinates), 3):
				start, end, separator = coordinates[index:index + 3]
				self.assertIsNone(separator[0])
				adjacency.setdefault(start, set()).add(end)
				adjacency.setdefault(end, set()).add(start)

		self.assertEqual(len(adjacency), 4**3 + 2)
		reached = set()
		pending = [next(iter(adjacency))]
		while pending:
			vertex = pending.pop()
			if vertex in reached:
				continue
			reached.add(vertex)
			pending.extend(adjacency[vertex] - reached)
		self.assertEqual(reached, set(adjacency))

	def test_renderer_hides_originally_labeled_particles_until_cell_click(self):
		with patch(
			"visualize.render_3d.go.Figure.write_html",
			autospec=True,
		) as write_html:
			render_lattice(_graph_data(), "unused.html", seed=42)

		figure = write_html.call_args.args[0]
		self.assertEqual(
			[trace.name for trace in figure.data[:5]],
			[
				"Simple cubic bonds",
				"BCC body-center bonds",
				"Shared lattice vertices",
				"SC cells (click to expand)",
				"BCC centers (click to expand)",
			],
		)
		self.assertIn("3 x 3 x 3", figure.layout.title.text)
		particle_traces = [
			trace
			for trace in figure.data
			if trace.name.startswith("Topology particles ")
		]
		self.assertEqual(len(particle_traces), 4)
		self.assertTrue(all(trace.visible is False for trace in particle_traces))
		self.assertTrue(all(trace.meta for trace in particle_traces))
		labels = {
			label
			for trace in particle_traces
			for label in trace.text
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
		post_script = write_html.call_args.kwargs["post_script"]
		self.assertIn("plotly_click", post_script)
		self.assertIn("Plotly.restyle", post_script)
		self.assertIn("isExpandedParticle", post_script)


if __name__ == "__main__":
	unittest.main()
