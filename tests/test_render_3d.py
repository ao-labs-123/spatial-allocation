import unittest

from visualize.render_3d import _outer_lattice_bond_trace


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

		self.assertEqual(trace.name, "Outer crystal lattice bonds")
		self.assertEqual(len(trace.x), 54 * 3)
		self.assertEqual(trace.x.count(None), 54)
		self.assertEqual(trace.y.count(None), 54)
		self.assertEqual(trace.z.count(None), 54)

	def test_returns_no_trace_without_unit_cell(self):
		self.assertIsNone(_outer_lattice_bond_trace({}, [(0, 0, 0)]))


if __name__ == "__main__":
	unittest.main()
