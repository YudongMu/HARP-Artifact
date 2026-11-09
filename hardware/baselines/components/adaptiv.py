"""AdapTME sign matching, token merges and reduced matrix tasks."""

from ...component_timing import Plan, Stage
from .common import LINEAR_ROLES, elements, engine, run_component


def phases(graph, profile):
    retained = profile["token_fraction"]

    def plan(node, layer, lanes):
        role = node["role"]
        if role == "self_attention":
            return engine(node, lanes, "AdapTiV_PE_Array", "AdapTiV_VPU", retained**2)
        if role in LINEAR_ROLES:
            main = engine(node, lanes, "AdapTiV_PE_Array", "AdapTiV_VPU", retained)
            if role not in {"qkv", "packed_in"}:
                return main
            values = elements(layer, node["inputs"][0], lanes)
            return Plan((
                Stage("AdapTME_SPMU", "scratchpad", values),
                Stage("AdapTME_SSCU", "sign_compare", values),
                Stage("AdapTME_token_merge", "merge", values),
                *main.serial,
            ))
        return engine(node, lanes, "AdapTiV_PE_Array", "AdapTiV_VPU")
    return [("trajectory", graph["forward_calls"], plan)]



if __name__ == "__main__":
    run_component("adaptiv", phases)
