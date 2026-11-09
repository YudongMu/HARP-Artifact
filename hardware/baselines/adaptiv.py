"""Run the adaptiv component model."""

from .components.adaptiv import phases
from .components.common import run_component


if __name__ == "__main__":
    run_component("adaptiv", phases)
