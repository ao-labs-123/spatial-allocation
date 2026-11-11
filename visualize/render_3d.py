import itertools
import json
import math
import random
import sys
from copy import deepcopy
from pathlib import Path

import plotly.graph_objects as go

if __package__ in (None, ""):
	sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.allocator import allocate_coordinates


MIN_CELLS_PER_AXIS = 3


def _site_key(kind, coordinate):
	return kind + ":" + ",".join(f"{value:g}" for value in coordinate)


def _allocate_components(graph_data, seed):
	allocated = allocate_coordinates(deepcopy(graph_data))
	nodes = {
		node["id"]: node
		for node in allocated["nodes"]
		if node.get("node_type") != "container"
	}
	containers = [
		node for node in allocated["nodes"]
		if node.get("container_type") == "cluster"
	]
	rng = random.Random(seed)
	rng.shuffle(containers)
	cells_per_axis = max(
		MIN_CELLS_PER_AXIS,
		math.ceil(len(containers) ** (1 / 3)),
	)
	cells = list(itertools.product(range(cells_per_axis), repeat=3))
	rng.shuffle(cells)
	cell_groups = {
		cells[index]: container
		for index, container in enumerate(containers)
	}
	return nodes, cell_groups, allocated["edges"], cells_per_axis


def _component_site_positions(members, cell):
	corners = list(
		itertools.product(*(range(value, value + 2) for value in cell))
	)
	node_coordinates = {
		node["id"]: tuple(float(value) for value in node["lattice_coordinate"])
		for node in members
	}
	vertex_nodes = [
		node for node in members
		if node.get("lattice_site") != "body_center"
	]
	vertex_nodes.sort(
		key=lambda node: (node_coordinates[node["id"]], node["id"])
	)
	center_nodes = [
		node for node in members
		if node.get("lattice_site") == "body_center"
	]
	center_nodes.sort(key=lambda node: node["id"])

	positions = {}
	for index, node in enumerate(vertex_nodes):
		coordinate = corners[index % len(corners)]
		repeated = index // len(corners)
		offset = ((repeated + 1) // 2) * (0.08 if repeated % 2 == 0 else -0.08)
		positions[node["id"]] = {
			"kind": "vertex",
			"coordinate": coordinate,
			"position": [float(value) for value in coordinate],
			"label_offset": offset,
		}

	center_coordinate = tuple(value + 0.5 for value in cell)
	for index, node in enumerate(center_nodes):
		positions[node["id"]] = {
			"kind": "center",
			"coordinate": center_coordinate,
			"position": [float(value) for value in center_coordinate],
			"label_offset": (index - (len(center_nodes) - 1) / 2) * 0.08,
		}
	return positions


def _mixed_lattice_geometry(graph_data, seed=None):
	nodes_by_id, cell_groups, edges, cells_per_axis = _allocate_components(
		graph_data,
		seed,
	)
	sc_x = []
	sc_y = []
	sc_z = []
	bcc_x = []
	bcc_y = []
	bcc_z = []

	def add_segment(x_values, y_values, z_values, start, end):
		x_values.extend([start[0], end[0], None])
		y_values.extend([start[1], end[1], None])
		z_values.extend([start[2], end[2], None])

	for vertex in itertools.product(range(cells_per_axis + 1), repeat=3):
		for axis in range(3):
			neighbor = list(vertex)
			neighbor[axis] += 1
			if neighbor[axis] <= cells_per_axis:
				add_segment(sc_x, sc_y, sc_z, vertex, neighbor)

	bcc_cells = {
		cell
		for cell, container in cell_groups.items()
		if container["lattice_mode"] == "body_centered_cubic"
	}
	for cell in bcc_cells:
		center = tuple(coordinate + 0.5 for coordinate in cell)
		for corner in itertools.product((0, 1), repeat=3):
			vertex = tuple(cell[axis] + corner[axis] for axis in range(3))
			add_segment(bcc_x, bcc_y, bcc_z, center, vertex)

	lattice_vertices = list(
		itertools.product(range(cells_per_axis + 1), repeat=3)
	)
	bcc_centers = [
		tuple(coordinate + 0.5 for coordinate in cell)
		for cell in sorted(bcc_cells)
	]
	site_contents = {}
	node_positions = {}
	for cell, container in cell_groups.items():
		members = [
			nodes_by_id[node_id]
			for node_id in container["contains"]
		]
		positions = _component_site_positions(members, cell)
		for node in members:
			placement = positions[node["id"]]
			key = _site_key(placement["kind"], placement["coordinate"])
			site_contents.setdefault(key, []).append(node)
			node_positions[node["id"]] = placement["position"]

	vertex_contents = {
		_site_key("vertex", vertex): site_contents.get(
			_site_key("vertex", vertex), []
		)
		for vertex in lattice_vertices
	}
	center_contents = {
		_site_key("center", center): site_contents.get(
			_site_key("center", center), []
		)
		for center in bcc_centers
	}
	all_sites = {**vertex_contents, **center_contents}
	site_graph_data = {}
	for key, site_nodes in all_sites.items():
		if not site_nodes:
			continue
		site_node_ids = {node["id"] for node in site_nodes}
		site_edges = [
			edge
			for edge in edges
			if edge["source"] in site_node_ids or edge["target"] in site_node_ids
		]
		label_by_id = {
			node["id"]: str(node.get("label", node["id"]))
			for node in nodes_by_id.values()
		}
		site_graph_data[key] = {
			"kind": "BCC center" if key.startswith("center:") else "Shared lattice vertex",
			"nodes": [
				{
					"id": node["id"],
					"label": str(node.get("label", node["id"])),
					"category": str(node.get("category", "")),
					"lattice_role": str(node.get("lattice_role", "")),
					"lattice_index": str(node.get("lattice_index", "")),
				}
				for node in site_nodes
			],
			"edges": [
				{
					"source": label_by_id[edge["source"]],
					"target": label_by_id[edge["target"]],
					"type": str(edge.get("morphism_type", "Relation")),
				}
				for edge in site_edges
			],
		}

	def site_hover(kind, coordinate):
		key = _site_key(kind, coordinate)
		count = len(all_sites.get(key, []))
		return f"{kind.title()}<br>{count} nested graph particles<br>Click to expand"

	geometry = [
		go.Scatter3d(
			x=sc_x,
			y=sc_y,
			z=sc_z,
			mode="lines",
			line={"color": "rgba(91, 107, 121, 0.88)", "width": 5},
			hoverinfo="skip",
			name="Simple cubic bonds",
		),
		go.Scatter3d(
			x=bcc_x,
			y=bcc_y,
			z=bcc_z,
			mode="lines",
			line={"color": "rgba(221, 130, 48, 0.95)", "width": 5},
			hoverinfo="skip",
			name="BCC body-center bonds",
		),
		go.Scatter3d(
			x=[point[0] for point in lattice_vertices],
			y=[point[1] for point in lattice_vertices],
			z=[point[2] for point in lattice_vertices],
			mode="markers",
			marker={
				"size": 8,
				"color": "rgba(65, 78, 91, 0.95)",
				"line": {"color": "white", "width": 1},
			},
			customdata=[_site_key("vertex", point) for point in lattice_vertices],
			text=[site_hover("vertex", point) for point in lattice_vertices],
			hovertemplate="%{text}<extra></extra>",
			name="Shared lattice vertices",
		),
		go.Scatter3d(
			x=[point[0] for point in bcc_centers],
			y=[point[1] for point in bcc_centers],
			z=[point[2] for point in bcc_centers],
			mode="markers",
			marker={
				"size": 11,
				"symbol": "diamond",
				"color": "rgba(239, 157, 69, 1)",
				"line": {"color": "white", "width": 1},
			},
			customdata=[_site_key("center", point) for point in bcc_centers],
			text=[site_hover("center", point) for point in bcc_centers],
			hovertemplate="%{text}<extra></extra>",
			name="BCC centers",
		),
	]

	return {
		"traces": geometry,
		"site_graph_data": site_graph_data,
		"cell_groups": cell_groups,
		"bcc_cells": bcc_cells,
		"cells_per_axis": cells_per_axis,
		"simple_cubic_bonds": geometry[0],
		"bcc_bonds": geometry[1],
		"vertices": geometry[2],
		"centers": geometry[3],
		"site_contents": all_sites,
	}


def render_lattice(graph_data, output_path, seed=None):
	geometry = _mixed_lattice_geometry(graph_data, seed)
	figure = go.Figure(data=geometry["traces"])

	post_script = """
var plot = document.getElementById('{plot_id}');
var siteData = __SITE_GRAPH_DATA__;
var panel = document.createElement('section');
panel.style.cssText = 'display:none;position:fixed;right:16px;top:64px;z-index:10000;width:min(380px,calc(100vw - 32px));max-height:70vh;overflow:auto;box-sizing:border-box;padding:16px;background:rgba(255,255,255,.97);border:1px solid #cbd5e1;border-radius:10px;box-shadow:0 8px 28px rgba(15,23,42,.22);font:14px/1.45 system-ui,sans-serif;color:#1e293b';
var panelHeader = document.createElement('div');
panelHeader.style.cssText = 'display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:12px';
var panelTitle = document.createElement('strong');
var closeButton = document.createElement('button');
closeButton.type = 'button';
closeButton.textContent = 'Close';
closeButton.style.cssText = 'padding:5px 9px;border:1px solid #cbd5e1;border-radius:6px;background:white;cursor:pointer';
var panelBody = document.createElement('div');
panelHeader.appendChild(panelTitle);
panelHeader.appendChild(closeButton);
panel.appendChild(panelHeader);
panel.appendChild(panelBody);
document.body.appendChild(panel);
closeButton.addEventListener('click', function() { panel.style.display = 'none'; });
plot.on('plotly_click', function(eventData) {
    var point = eventData.points && eventData.points[0];
    if (!point) return;
    var isLatticeSite = point.data.name === 'Shared lattice vertices' ||
                        point.data.name === 'BCC centers';
    if (!isLatticeSite) return;
    var siteKey = point.customdata;
    var data = siteData[siteKey];
    if (!data || !data.nodes.length) return;
    panelTitle.textContent = data.kind + ' [' + siteKey.split(':')[1] + ']';
    panelBody.replaceChildren();
    var particleHeading = document.createElement('div');
    particleHeading.textContent = 'Nested graph particles (' + data.nodes.length + ')';
    particleHeading.style.cssText = 'font-weight:600;margin:8px 0';
    panelBody.appendChild(particleHeading);
    data.nodes.forEach(function(node) {
        var item = document.createElement('div');
        item.style.cssText = 'padding:7px 9px;margin:5px 0;background:#f1f5f9;border-radius:6px;overflow-wrap:anywhere';
        var label = document.createElement('div');
        label.textContent = node.label;
        label.style.fontWeight = '600';
        item.appendChild(label);
        var detail = document.createElement('div');
        detail.textContent = node.category +
            (node.lattice_role ? ' · ' + node.lattice_role : '') +
            (node.lattice_index ? ' · index: ' + node.lattice_index : '');
        detail.style.cssText = 'color:#64748b;font-size:12px';
        item.appendChild(detail);
        panelBody.appendChild(item);
    });
    if (data.edges.length) {
        var edgeHeading = document.createElement('div');
        edgeHeading.textContent = 'Connections (' + data.edges.length + ')';
        edgeHeading.style.cssText = 'font-weight:600;margin:14px 0 8px';
        panelBody.appendChild(edgeHeading);
        data.edges.forEach(function(edge) {
            var item = document.createElement('div');
            item.textContent = edge.source + ' —' + edge.type + '→ ' + edge.target;
            item.style.cssText = 'padding:5px 0;border-top:1px solid #e2e8f0;overflow-wrap:anywhere';
            panelBody.appendChild(item);
        });
    }
    panel.style.display = 'block';
});
""".replace("__SITE_GRAPH_DATA__", json.dumps(geometry["site_graph_data"]))

	cells_per_axis = geometry["cells_per_axis"]
	figure.update_layout(
		title=(
			"Mixed SC / BCC crystal lattice with expandable graph capsules "
			f"({cells_per_axis} x {cells_per_axis} x {cells_per_axis})"
		),
		margin={"l": 0, "r": 0, "t": 48, "b": 0},
		legend={"title": "Crystal lattice"},
		hovermode="closest",
		scene={
			"xaxis": {"title": "X", "range": [-0.25, cells_per_axis + 0.25]},
			"yaxis": {"title": "Y", "range": [-0.25, cells_per_axis + 0.25]},
			"zaxis": {"title": "Z", "range": [-0.25, cells_per_axis + 0.25]},
			"aspectmode": "data",
			"dragmode": "orbit",
			"camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.2}},
		},
	)
	output_path = Path(output_path)
	figure.write_html(
		output_path,
		include_plotlyjs=True,
		full_html=True,
		post_script=post_script,
	)
	if output_path.exists():
		html_lines = output_path.read_text(encoding="utf-8").splitlines()
		output_path.write_text(
			"\n".join(line.rstrip() for line in html_lines) + "\n",
			encoding="utf-8",
		)


def main():
	project_dir = Path(__file__).resolve().parents[1]
	input_path = project_dir / "data" / "topological_graph.json"
	output_path = project_dir / "docs" / "index.html"
	with input_path.open(encoding="utf-8") as input_file:
		graph_data = json.load(input_file)
	output_path.parent.mkdir(parents=True, exist_ok=True)
	render_lattice(graph_data, output_path)
	print(f"Saved expandable mixed crystal lattice visualization to {output_path}")


if __name__ == "__main__":
	main()
