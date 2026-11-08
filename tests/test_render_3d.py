import unittest

from visualize.render_3d import _inner_lattice_geometry, _outer_lattice_bond_trace


class OuterLatticeBondTraceTests(unittest.TestCase):
	def test_connects_adjacent_repeated_unit_cells(self):
		crystal = {
			"unit_cell": {
				"center": [1.0, 2.0, 3.0],
				"size": [4.0, 5.0, 6.0],
			},
		}
		cell_coordinates = [
			(x, y, z)
			for x in range(-1, 2)
			for y in range(-1, 2)
			for z in range(-1, 2)
		]

		trace = _outer_lattice_bond_trace(crystal, cell_coordinates)

		self.assertEqual(trace.name, "Supercell repeat connections")
		self.assertEqual(len(trace.x), 54 * 3)
		self.assertEqual(trace.x.count(None), 54)
		self.assertEqual(trace.y.count(None), 54)
		self.assertEqual(trace.z.count(None), 54)

	def test_returns_no_trace_without_unit_cell(self):
		self.assertIsNone(_outer_lattice_bond_trace({}, [(0, 0, 0)]))


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

	def test_simple_cubic_draws_occupied_and_empty_vertices_and_bonds(self):
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
		self.assertEqual(sum("Occupied" in value for value in vertices.customdata), 1)
		self.assertEqual(sum("Empty" in value for value in vertices.customdata), 7)
		self.assertEqual(len(centers.x), 0)
		self.assertEqual(positions[particle["_render_id"]], [-0.72, -0.62, -0.62])

	def test_body_centered_cubic_draws_occupied_and_empty_sites_and_bonds(self):
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
		self.assertIn("Occupied", centers.customdata[0])
		self.assertEqual(positions["core@0,0,0"], [0.0, 0.0, 0.0])


if __name__ == "__main__":
	unittest.main()
