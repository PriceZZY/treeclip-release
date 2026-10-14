"""Weiser 2022 (SYSSIFOSS) ALS dataset layout.

Reference: Weiser et al., 2022, ESSD 14(7), 2989-3012; data at PANGAEA.942856.

Plots with species labels (6 of 12): BR01, BR03, BR05, KA09, KA10, KA11.
Plots without species: BR02, BR04, BR06, BR07, BR08, SP02 (LiDAR only).

File naming convention (per metadata PDF section 3.1):
    Single-tree point cloud filename:
        SPECIESID_PLOTID_TREEID_YYYY-MM-DD(_SUFFIX).laz
    e.g.  FagSyl_BR01_t001_2019-07-05.laz

    SPECIESID = first 3 letters of genus + first 3 letters of species epithet
                e.g.  FagSyl = Fagus sylvatica, QueRob = Quercus robur
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

# Plots with field-inventory species labels
LABELED_PLOTS = ("BR01", "BR03", "BR05", "KA09", "KA10", "KA11")
ALL_PLOTS = (
    "BR01", "BR02", "BR03", "BR04", "BR05", "BR06", "BR07", "BR08",
    "KA09", "KA10", "KA11", "SP02",
)

# Re-pattern for parsing a single-tree LAS filename. Real examples (KA10):
#   PinSyl_KA10_01_2019-07-05_q2_ALS-on.laz
#   AcePse_KA10_P10Tree13_2019-07-05_q4_ALS-on.laz
#   CarBet_KA10_P13Tree10_2019-07-05_q4_ALS-on.laz
# SPECIES = 6-letter biological code (3-letter genus + 3-letter species epithet)
SINGLE_TREE_PATTERN = re.compile(
    r"^(?P<species>[A-Z][a-z]+[A-Z][a-z]+)_"
    r"(?P<plot>[A-Z]{2}\d{2})_"
    r"(?P<treeid>[^_]+)_"
    r"(?P<date>\d{4}-\d{2}-\d{2})"
    r"(?:_(?P<suffix>[^.]+))?"
    r"\.la[sz]$",
)


@dataclass(frozen=True)
class WeiserLayout:
    """Expected on-disk layout after unpacking each plot's zip.

    Each plot zip extracts to {root}/{plot_id}/ containing:
      single_trees/{platform}-{canopy}/{SPECIES}_{PLOT}_{TREEID}_{DATE}.laz
        platform ∈ {ALS, ULS, TLS}; canopy ∈ {on, off}
      (plus raw plot-level scans and trajectories — not needed for our work)
    """
    root: Path

    def plot_dir(self, plot: str) -> Path:
        self._check_plot(plot)
        return self.root / plot

    def single_trees_dir(self, plot: str) -> Path:
        """Directory containing the extracted per-tree point clouds for one plot."""
        return self.plot_dir(plot) / "single_trees"

    def _check_plot(self, plot: str) -> None:
        if plot not in ALL_PLOTS:
            raise ValueError(f"Unknown plot {plot!r}; expected one of {ALL_PLOTS}")


def parse_single_tree_filename(filename: str) -> dict | None:
    """Parse a single-tree filename. Returns dict with species/plot/treeid/date, or None."""
    m = SINGLE_TREE_PATTERN.match(Path(filename).name)
    return m.groupdict() if m else None
