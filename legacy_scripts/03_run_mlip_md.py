from opa_ketone_workflow import (
    WorkflowInput,
    load_workflow_input,
    prepare_structure_input,
    run_mlip_md,
)


try:
    config = load_workflow_input("workflow_input.json")
except FileNotFoundError:
    config = WorkflowInput()

prepare_structure_input(config)

# Define your MLIP calculator here.
#
# Example: MACE
# from mace.calculators import MACECalculator
# calc = MACECalculator(
#     model_paths="mace_model.model",
#     device="cuda",
#     default_dtype="float64",
# )
#
# Example: Matlantis
# from pfp_api_client.pfp.estimator import Estimator
# from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
# estimator = Estimator(calc_mode="CRYSTAL_U0")
# calc = ASECalculator(estimator)
#
# Example: DeepMD
# from deepmd.calculator import DP
# calc = DP(model="frozen_model.pb")

raise RuntimeError("Define your MLIP calculator as `calc`, then remove this line.")

run_mlip_md(config, calc)
