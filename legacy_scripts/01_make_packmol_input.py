from opa_ketone_workflow import (
    WorkflowInput,
    prepare_structure_input,
    save_workflow_input,
    write_packmol_input,
)


config = WorkflowInput()
prepare_structure_input(config)
save_workflow_input(config)
write_packmol_input(config)
