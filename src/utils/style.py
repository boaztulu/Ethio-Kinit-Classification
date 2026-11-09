"""Matplotlib style helpers for consistent publication figures."""
from pathlib import Path
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STYLE_PATH = PROJECT_ROOT / "configs" / "matplotlib_style.mplstyle"

CLASS_NAMES = ["Tizita", "Bati", "Ambassel", "Anchihoye"]
CLASS_COLORS = {
    "Tizita":    "#1f77b4",
    "Bati":      "#ff7f0e",
    "Ambassel":  "#2ca02c",
    "Anchihoye": "#d62728",
}


def apply_style() -> None:
    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))
