"""DSTAR FMDQ QPU, SAR and mixed-precision MPU services."""

from math import ceil

from ...component_timing import Plan, Stage
from .common import elements, engine, temporal_phases, run_component


def phases(graph, profile):
    def build(initial, final):
        def plan(node, layer, lanes):
            role = node["role"]
            if role == "self_attention":
                main = engine(node, lanes, "DSTAR_MPU_SAR", "DSTAR_VPU_softmax",
                              profile["sar_work_fraction"])
                if node["kind"] != "qk":
                    return main
                score_tiles = ceil(node["attributes"]["pair_elements"] / 256)
                return Plan((Stage("DSTAR_SAR_index", "sar", score_tiles * lanes), *main.serial))
            if role not in {"ffn_in", "ffn_out"}:
                main = engine(node, lanes, "DSTAR_MPU_A8W8", "DSTAR_VPU")
                return Plan(main.serial, weight_bytes=node["weight_bytes"] // 2)
            values = elements(layer, node["output"], lanes)
            weights = node["weight_bytes"] // 2
            if initial:
                seed = () if final else (Stage("DSTAR_FMDQ_cache_seed", "hbm", 2 * values),)
                return Plan((Stage("DSTAR_MPU_A8W8", "pe", node["macs"] * lanes),), join=seed, weight_bytes=weights)
            input_values = elements(layer, node["inputs"][0], lanes)
            return Plan((
                Stage("DSTAR_QPU_FMDQ", "qpu", input_values),
                Stage("DSTAR_MPU_FMDQ", "pe", ceil(node["macs"] * lanes * profile["fmdq_bits"] / 8)),
            ), (Stage("DSTAR_FMDQ_cache_read", "hbm", 2 * values),), (
                Stage("DSTAR_VPU_reverse_difference", "vector", values),
                *((Stage("DSTAR_FMDQ_cache_write", "hbm", 2 * values),) if not final else ()),
            ), weight_bytes=weights)
        return plan
    return temporal_phases(graph, build)



if __name__ == "__main__":
    run_component("dstar", phases)
