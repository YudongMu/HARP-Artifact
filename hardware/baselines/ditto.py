"""Run the ditto component model."""

from .components.ditto import phases
from .components.common import run_component


if __name__ == "__main__":
    run_component("ditto", phases)
