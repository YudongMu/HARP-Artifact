"""Run the dstar component model."""

from .components.dstar import phases
from .components.common import run_component


ARCHITECTURE = {
    "clock_hz": 1_000_000_000,
    "macs_per_cycle": 32 * 64 * 64,
    "vector_ops_per_cycle": 32 * 64,
    "hbm_bytes_per_cycle": 1500,
}


if __name__ == "__main__":
    run_component("dstar", phases)
