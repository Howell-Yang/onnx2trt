import onnx


def find_common_input(onnx_model, node):
    pass

def fuse_repconvs(onnx_model):
    # Step 01: find rep convs
    for node in onnx_model.graph.node:
        if node.op_type == "Add":
            find_common_input(onnx_model, node)


    # step 02: merge rep conv weights


    # step 03: remove extra conv and bn nodes