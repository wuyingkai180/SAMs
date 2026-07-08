from opa_ketone_workflow import (
    WorkflowInput,
    analyze_opa_motion,
    load_workflow_input,
    plot_opa_motion,
    prepare_structure_input,
)


try:
    config = load_workflow_input("workflow_input.json")
except FileNotFoundError:
    config = WorkflowInput()

prepare_structure_input(config)
df = analyze_opa_motion(config)
plot_opa_motion(config, df)
