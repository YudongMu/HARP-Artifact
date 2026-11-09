"""BitRed bit-serial issue and its activation dispatch and reduction services."""

from math import ceil

from ...component_timing import Plan, Stage
from .common import LINEAR_ROLES, elements, engine, run_component


def phases(graph, profile):
    def plan(node, layer, lanes):
        if node["role"] not in LINEAR_ROLES:
            return engine(node, lanes, "BitRed_array", "BitRed_vector")
        activations = elements(layer, node["inputs"][0], lanes)
        outputs = elements(layer, node["output"], lanes)
        return Plan((
            Stage("BitRed_cal.adis", "dispatch", activations),
            Stage("BitRed_bit_serial_array", "pe",
                  ceil(node["macs"] * lanes * profile["linear_issue_fraction"])),
            Stage("BitRed_cal.red", "vector", outputs),
        ))
    return [("trajectory", graph["forward_calls"], plan)]



if __name__ == "__main__":
    run_component("bitred", phases)
