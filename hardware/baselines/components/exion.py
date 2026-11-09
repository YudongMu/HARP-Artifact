"""EXION ConMerge, Eager Prediction Engine and DSC."""

from math import ceil

from ...component_timing import Plan, Stage
from .common import elements, engine, temporal_phases, run_component


def phases(graph, profile):
    def build(initial, final):
        def plan(node, layer, lanes):
            role = node["role"]
            if role == "self_attention":
                fraction = profile["attention_fraction"]
                prefix = ((Stage("EXION_EPRE", "epre", node["attributes"]["pair_elements"] *
                                 node["attributes"]["heads"] * lanes),) if node["kind"] == "qk" else ())
                main = engine(node, lanes, "EXION_DSC", "EXION_vector", fraction)
                return Plan(prefix + main.serial)
            if role not in {"ffn_in", "ffn_out"} or initial:
                main = engine(node, lanes, "EXION_DSC", "EXION_vector")
                if role != "ffn_in" or final:
                    return main
                size = layer["tensors"][node["output"]]["bytes"] * lanes
                return Plan(main.serial, join=(Stage("EXION_FFN_cache_seed", "hbm", size),))
            values = elements(layer, node["output"], lanes)
            main = (
                Stage("EXION_ConMerge", "conmerge", values),
                Stage("EXION_DSC_compacted", "pe", ceil(node["macs"] * lanes * profile["ffn_fraction"])),
            )
            if role == "ffn_out":
                return Plan(main)
            return Plan(main, (Stage("EXION_FFN_cache_read", "hbm", 2 * values),),
                        () if final else (Stage("EXION_FFN_cache_write", "hbm", 2 * values),))
        return plan
    return temporal_phases(graph, build)



if __name__ == "__main__":
    run_component("exion", phases)
