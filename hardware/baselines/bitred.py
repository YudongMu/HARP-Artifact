"""Run the bitred component model."""

from .components.bitred import phases
from .components.common import run_component


if __name__ == "__main__":
    run_component("bitred", phases)
