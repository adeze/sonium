#!/usr/bin/env python3
"""Validate the XEM external-format release fixture with independent readers.

ElementTree is intentionally separate from Rust's `vtkio`; h5py validates the
SOFA HDF5 hierarchy without using `sofa-reader`.  This script is a required
release-environment capability check, not a replacement for ParaView UI smoke
coverage.
"""

from __future__ import annotations

import math
import pathlib
import sys
import xml.etree.ElementTree as element_tree

try:
    import h5py
except ImportError as error:
    raise SystemExit(
        "h5py is required for the independent SOFA reader smoke; "
        "install it in the release-validation environment"
    ) from error


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def children(element: element_tree.Element, name: str) -> list[element_tree.Element]:
    return [child for child in element if local_name(child.tag) == name]


def only_child(element: element_tree.Element, name: str) -> element_tree.Element:
    found = children(element, name)
    if len(found) != 1:
        raise ValueError(f"expected exactly one {name}, found {len(found)}")
    return found[0]


def check_vtu(path: pathlib.Path) -> None:
    root = element_tree.parse(path).getroot()
    if local_name(root.tag) != "VTKFile" or root.attrib.get("type") != "UnstructuredGrid":
        raise ValueError("not an UnstructuredGrid VTU document")
    grid = only_child(root, "UnstructuredGrid")
    piece = only_child(grid, "Piece")
    if piece.attrib.get("NumberOfPoints") != "4" or piece.attrib.get("NumberOfCells") != "2":
        raise ValueError("unexpected VTU mesh dimensions")
    point_data = only_child(piece, "PointData")
    cell_data = only_child(piece, "CellData")
    point_names = {array.attrib.get("Name") for array in children(point_data, "DataArray")}
    cell_names = {array.attrib.get("Name") for array in children(cell_data, "DataArray")}
    if "temperature" not in point_names or "region" not in cell_names:
        raise ValueError("expected point/cell arrays are missing")
    cells = only_child(piece, "Cells")
    cell_arrays = {array.attrib.get("Name") for array in children(cells, "DataArray")}
    if not {"connectivity", "offsets", "types"}.issubset(cell_arrays):
        raise ValueError("VTU topology arrays are missing")


def check_pvd(path: pathlib.Path) -> None:
    root = element_tree.parse(path).getroot()
    if local_name(root.tag) != "VTKFile" or root.attrib.get("type") != "Collection":
        raise ValueError("not a PVD collection")
    collection = only_child(root, "Collection")
    datasets = children(collection, "DataSet")
    if len(datasets) != 1:
        raise ValueError("expected one PVD dataset")
    dataset = datasets[0]
    if dataset.attrib.get("file") != "frame_000.vtu":
        raise ValueError("PVD does not reference the fixture VTU")
    if not math.isclose(float(dataset.attrib["timestep"]), 0.0, abs_tol=0.0):
        raise ValueError("unexpected PVD timestep")


def check_sofa(path: pathlib.Path) -> None:
    with h5py.File(path, "r") as sofa:
        convention = sofa.attrs.get("SOFAConventions")
        if isinstance(convention, bytes):
            convention = convention.decode("utf-8")
        if convention != "SimpleFreeFieldHRTF":
            raise ValueError("unexpected SOFA convention")
        if sofa["Data.Real"].shape != (2, 1, 2):
            raise ValueError("unexpected Data.Real shape")
        if sofa["Data.Imag"].shape != (2, 1, 2):
            raise ValueError("unexpected Data.Imag shape")
        if sofa["Data.Real"][()].flatten().tolist() != [1.0, 2.0, 3.0, 4.0]:
            raise ValueError("unexpected Data.Real values")
        if sofa["Data.Imag"][()].flatten().tolist() != [0.1, 0.2, 0.3, 0.4]:
            raise ValueError("unexpected Data.Imag values")
        if list(sofa["N"][()]) != [500.0, 1000.0]:
            raise ValueError("unexpected SOFA frequency axis")


def main(directory: pathlib.Path) -> None:
    check_vtu(directory / "frame_000.vtu")
    check_pvd(directory / "series.pvd")
    check_sofa(directory / "minimal.sofa")
    print("external XEM VTU/PVD/SOFA reader smoke passed")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: read_external_formats.py <fixture-directory>")
    main(pathlib.Path(sys.argv[1]))
