"""Computer-vision subtasks, each usable and inspectable on its own.

The full reconstruction lives in :mod:`miocard.pipeline`. This package holds the
steps that are useful in isolation - measuring the grid, flattening the sheet
against it, turning a sheet into a digital code and drawing that code back as an
image - so each can be run, looked at and judged without the rest.
"""

from __future__ import annotations

from miocard.cv.rectify import (
    LatticeField,
    Rectified,
    RectifyConfig,
    measure_lattice_field,
    rectify_to_grid,
)

__all__ = [
    "LatticeField",
    "RectifyConfig",
    "Rectified",
    "measure_lattice_field",
    "rectify_to_grid",
]
