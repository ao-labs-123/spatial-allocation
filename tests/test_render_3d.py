import itertools
import unittest
from unittest.mock import patch

from visualize.render_3d import CELLS_PER_AXIS, _mixed_lattice_geometry, render_lattice


class MixedLatticeGeometryTests(unittest.TestCase):
	def test_three_by_three_by_three_cells_have_a_random_mixed_assignment(self):
		first = _mixed_lattice_geometry(seed=42)
		second = _mixed_lattice_geometry(seed=42)
		cells = set(first["bcc_cells"])
		ordered_cells = list(itertools.product(range(CELLS_PER_AXIS), repeat=3))
		all_cells = set(ordered_cells)

		self.assertEqual(CELLS_PER_AXIS, 3)
		self.assertEqual(len(all_cells), 27)
		self.assertEqual(len(cells), len(all_cells) // 2)
		self.assertEqual(first["bcc_cells"], second["bcc_cells"])
		self.assertEqual(len(all_cells - cells), len(all_cells) - len(all_cells) // 2)
		self.assertNotEqual(cells, set(ordered_cells[:4]))

	def test_all_simple_cubic_and_bcc_bonds_form_one_connected_lattice(self):
		geometry = _mixed_lattice_geometry(seed=42)
		traces = [geometry["simple_cubic_bonds"], geometry["bcc_bonds"]]
		adjacency = {}

		for trace in traces:
			coordinates = list(zip(trace.x, trace.y, trace.z))
			for index in range(0, len(coordinates), 3):
				start, end, separator = coordinates[index:index + 3]
				self.assertIsNone(separator[0])
				adjacency.setdefault(start, set()).add(end)
				adjacency.setdefault(end, set()).add(start)

		sc_edge_count = 3 * CELLS_PER_AXIS * (CELLS_PER_AXIS + 1) ** 2
		self.assertEqual(len(geometry["simple_cubic_bonds"].x), sc_edge_count * 3)
		self.assertEqual(len(geometry["bcc_bonds"].x), len(geometry["bcc_cells"]) * 8 * 3)
		self.assertEqual(len(adjacency), (CELLS_PER_AXIS + 1) ** 3 + len(geometry["bcc_cells"]))

		reached = set()
		pending = [next(iter(adjacency))]
		while pending:
			vertex = pending.pop()
			if vertex in reached:
				continue
			reached.add(vertex)
			pending.extend(adjacency[vertex] - reached)
		self.assertEqual(reached, set(adjacency))

	def test_renderer_shows_only_the_mixed_crystal_lattice(self):
		with patch(
			"visualize.render_3d.go.Figure.write_html",
			autospec=True,
		) as write_html:
			render_lattice("unused.html", seed=42)

		figure = write_html.call_args.args[0]
		self.assertEqual(
			[trace.name for trace in figure.data],
			[
				"Simple cubic bonds",
				"BCC body-center bonds",
				"Shared lattice vertices",
				"BCC body-center atoms",
			],
		)
		self.assertEqual(len(figure.data[2].x), 64)
		self.assertEqual(len(figure.data[3].x), 13)
		self.assertIn("3 x 3 x 3", figure.layout.title.text)


if __name__ == "__main__":
	unittest.main()
