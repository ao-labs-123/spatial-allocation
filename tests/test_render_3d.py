import unittest
from unittest.mock import patch

from visualize.render_3d import (
	_inner_lattice_geometry,
	_particle_marker_color,
	render_graph,
)


class EventParticleColorTests(unittest.TestCase):
	def test_unassigned_event_is_muted_and_assigned_event_keeps_category_color(self):
		category_color = "#4477aa"
		unassigned = {"category": "Event", "lattice_site": "vertex"}
		assigned = {"category": "Event", "lattice_role": "vertex"}

		self.assertEqual(
			_particle_marker_color(unassigned, category_color),
			"rgba(145, 153, 160, 0.9)",
		)
		self.assertEqual(_particle_marker_color(assigned, category_color), category_color)


class RenderGraphTests(unittest.TestCase):
	def test_supercell_lattice_shares_vertices_without_adding_particles(self):
		graph_data = {
			"nodes": [
				{
					"id": "entity",
					"label": "Researcher",
					"category": "Entity",
					"node_type": "particle",
					"parent_id": "cluster",
					"lattice_role": "core",
					"lattice_site": "body_center",
					"lattice_coordinate": [0.5, 0.5, 0.5],
					"position": [0.0, 0.0, 0.0],
				},
				{
					"id": "event",
					"label": "Reviewed",
					"category": "Event",
					"node_type": "particle",
					"parent_id": "cluster",
					"lattice_role": "vertex",
					"lattice_site": "vertex",
					"lattice_coordinate": [0.0, 0.0, 0.0],
					"position": [-0.72, -0.62, -0.62],
				},
				{
					"id": "cluster",
					"label": "Local cluster",
					"node_type": "container",
					"container_type": "cluster",
					"position": [0.0, 0.0, 0.0],
					"capsule_half_length": 0.5,
					"capsule_radius": 1.0,
					"lattice_mode": "body_centered_cubic",
					"contains": ["entity", "event"],
				},
			],
			"edges": [],
			"crystal": {
				"lattice_mode": "body_centered_cubic",
				"preview_repeats": [2, 2, 2],
				"unit_cell": {"center": [0.0, 0.0, 0.0], "size": [4.0, 4.0, 4.0]},
			},
		}

		with patch(
			"visualize.render_3d.go.Figure.write_html",
			autospec=True,
		) as write_html:
			render_graph(graph_data, "unused.html")

		figure = write_html.call_args.args[0]
		traces = {trace.name: trace for trace in figure.data}
		self.assertIn("Outer lattice sites", traces)
		self.assertIn("Outer BCC center sites", traces)
		self.assertIn("Supercell lattice bonds", traces)
		self.assertNotIn("Topology containers", traces)
		self.assertEqual(len(traces["Outer lattice sites"].x), 27)
		self.assertEqual(len(set(zip(
			traces["Outer lattice sites"].x,
			traces["Outer lattice sites"].y,
			traces["Outer lattice sites"].z,
		))), 27)
		self.assertEqual(len(traces["Outer BCC center sites"].x), 8)
		self.assertEqual(len(traces["Supercell lattice bonds"].x), 118 * 3)
		entity_trace = traces["Entity"]
		container_labels = traces["Topology container labels"]

		self.assertEqual(entity_trace.x[0], -4.0)
		self.assertEqual(entity_trace.y[0], -4.0)
		self.assertEqual(entity_trace.z[0], -4.0)
		self.assertEqual(container_labels.mode, "text")
		self.assertGreater(container_labels.z[0], entity_trace.z[0])


class InnerLatticeGeometryTests(unittest.TestCase):
	def _container(self, lattice_mode):
		return {
			"id": "cluster",
			"cell_coordinate": [0, 0, 0],
			"_render_id": "cluster@0,0,0",
			"lattice_mode": lattice_mode,
			"capsule_half_length": 1.0,
			"capsule_radius": 1.0,
		}

	def _particle(self, node_id, coordinate, lattice_site):
		return {
			"id": node_id,
			"topology_cluster_id": "cluster",
			"cell_coordinate": [0, 0, 0],
			"_render_id": f"{node_id}@0,0,0",
			"lattice_coordinate": list(coordinate),
			"lattice_site": lattice_site,
		}

	def test_simple_cubic_draws_uniform_lattice_vertices_and_bonds(self):
		container = self._container("simple_cubic")
		particle = self._particle("particle", (0, 0, 0), "vertex")
		positions = {
			container["_render_id"]: [0.0, 0.0, 0.0],
			particle["_render_id"]: [0.0, 0.0, 0.0],
		}

		bonds, vertices, centers = _inner_lattice_geometry(
			[container], [particle], positions
		)

		self.assertEqual(len(bonds.x), 12 * 3)
		self.assertEqual(len(vertices.x), 8)
		self.assertEqual(vertices.name, "Lattice vertices")
		self.assertEqual(set(vertices.customdata), {
			"Lattice vertex: 0, 0, 0",
			"Lattice vertex: 0, 0, 1",
			"Lattice vertex: 0, 1, 0",
			"Lattice vertex: 0, 1, 1",
			"Lattice vertex: 1, 0, 0",
			"Lattice vertex: 1, 0, 1",
			"Lattice vertex: 1, 1, 0",
			"Lattice vertex: 1, 1, 1",
		})
		self.assertEqual(vertices.marker.color, "rgba(145, 153, 160, 0.55)")
		self.assertEqual(len(centers.x), 0)
		self.assertEqual(positions[particle["_render_id"]], [-0.72, -0.62, -0.62])

	def test_body_centered_cubic_draws_uniform_sites_and_bonds(self):
		container = self._container("body_centered_cubic")
		particles = [
			self._particle("core", (0.5, 0.5, 0.5), "body_center"),
			self._particle("vertex", (0, 0, 0), "vertex"),
		]
		positions = {
			container["_render_id"]: [0.0, 0.0, 0.0],
			"core@0,0,0": [0.0, 0.0, 0.0],
			"vertex@0,0,0": [0.0, 0.0, 0.0],
		}

		bonds, vertices, centers = _inner_lattice_geometry(
			[container], particles, positions
		)

		self.assertEqual(len(bonds.x), (12 + 8) * 3)
		self.assertEqual(len(vertices.x), 8)
		self.assertEqual(len(centers.x), 1)
		self.assertEqual(centers.customdata[0], "BCC center: 0.5, 0.5, 0.5")
		self.assertEqual(centers.marker.color, "rgba(145, 153, 160, 0.55)")
		self.assertEqual(positions["core@0,0,0"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
	unittest.main()
