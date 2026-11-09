"""Run the exion component model."""

from .components.exion import phases
from .components.common import run_component


if __name__ == "__main__":
    run_component("exion", phases)
