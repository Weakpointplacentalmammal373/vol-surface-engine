"""Dashboard visualization components."""

from dashboard.components.arbitrage_diag import render_arbitrage_diagnostics
from dashboard.components.delta_smile import render_delta_smile
from dashboard.components.greeks import render_greeks
from dashboard.components.local_vol import render_local_vol
from dashboard.components.residual_heatmap import render_residual_heatmap
from dashboard.components.smile_slice import render_smile_slices
from dashboard.components.surface_3d import render_surface_3d
from dashboard.components.term_structure import render_mispricing_table, render_term_structure

__all__ = [
    "render_arbitrage_diagnostics",
    "render_delta_smile",
    "render_greeks",
    "render_local_vol",
    "render_mispricing_table",
    "render_residual_heatmap",
    "render_smile_slices",
    "render_surface_3d",
    "render_term_structure",
]
