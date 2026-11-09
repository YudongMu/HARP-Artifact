"""Ditto difference encoding, output cache and mixed-width issue."""

from math import ceil

from ...component_timing import Plan, Stage
from .common import LINEAR_ROLES, elements, engine, temporal_phases, run_component


def phases(graph, profile):
    def build(initial, final):
        def plan(node, layer, lanes):
            if node["role"] not in LINEAR_ROLES:
                return engine(node, lanes, "Ditto_PE", "Ditto_vector")
            work = node["macs"] * lanes
            output_bytes = layer["tensors"][node["output"]]["bytes"] * lanes
            weights = node["weight_bytes"] // 2
            if initial:
                store = () if final else (Stage("Ditto_cache_seed", "hbm", output_bytes),)
                return Plan((Stage("Ditto_PE_full_width", "pe", work),), join=store, weight_bytes=weights)
            activations = elements(layer, node["inputs"][0], lanes)
            return Plan((
                Stage("Ditto_Encoding_Unit", "encoding", activations),
                Stage("Ditto_PE_difference", "pe", ceil(work * profile["difference_issue_fraction"])),
            ), (Stage("Ditto_cached_output_read", "hbm", output_bytes),), (
                Stage("Ditto_output_restore", "vector", output_bytes // 2),
                *((Stage("Ditto_cached_output_write", "hbm", output_bytes),) if not final else ()),
            ), weight_bytes=weights)
        return plan
    return temporal_phases(graph, build)



if __name__ == "__main__":
    run_component("ditto", phases)
