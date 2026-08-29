"""Lower a fitted sklearn tree ensemble to STANDARD-domain ONNX ops.

Why this exists, measured: skl2onnx converts RandomForestRegressor to a single
`ai.onnx.ml.TreeEnsembleRegressor` node, and OpenVINO's ONNX frontend has no
conversion rule for the ai.onnx.ml domain — `core.read_model` fails with
"No conversion rule found for operations: ai.onnx.ml.TreeEnsembleRegressor"
(OpenVINO 2026.3). The MLP challenger only deploys because its graph is plain
MatMul/Add. So for tree ensembles we build the graph ourselves from ops
OpenVINO does execute: Gather, GatherElements, LessOrEqual, Where, ReduceMean.

The encoding: every tree's node arrays are padded to a common length and
flattened into one constant per attribute (feature id, threshold, left child,
right child, leaf value), with children stored as GLOBAL flat indices and
leaves pointing at themselves. Each sample starts at every tree's root and
the graph unrolls `max_depth` steps of

    node = where(x[feature[node]] <= threshold[node], left[node], right[node])

after which every walker sits on a (self-looping) leaf; the output is the
mean of the gathered leaf values — exactly RandomForestRegressor.predict.

Numeric fidelity: sklearn casts X to float32 but compares against float64
thresholds — and OpenVINO's CPU plugin executes float64 ops in float32, so a
naive double-precision graph misroutes samples sitting exactly on a split
(measured: max error 0.05 on a 120-tree forest). The graph therefore stays in
float32 and stores each threshold as the LARGEST float32 <= t: for float32
inputs, `x <= float32_floor(t)` is exactly equivalent to sklearn's
`float32(x) <= float64(t)`. Parity vs sklearn is asserted by the caller on
real data, not assumed here.
"""

from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OPSET = 18


def forest_to_onnx(forest, n_features: int, name: str = "forest") -> onnx.ModelProto:
    """Convert a fitted RandomForestRegressor / ExtraTreesRegressor (single
    output) to an ONNX model using only standard-domain ops.

    Input:  "X"  float32 [None, n_features]
    Output: "variable" float32 [None, 1]   (same contract as the skl2onnx MLP)
    """
    trees = [est.tree_ for est in forest.estimators_]
    n_trees = len(trees)
    k = max(t.node_count for t in trees)  # padded nodes per tree
    depth = max(int(t.max_depth) for t in trees)

    feat = np.zeros((n_trees, k), dtype=np.int64)
    thresh = np.zeros((n_trees, k), dtype=np.float64)
    left = np.zeros((n_trees, k), dtype=np.int64)
    right = np.zeros((n_trees, k), dtype=np.int64)
    value = np.zeros((n_trees, k), dtype=np.float32)

    for t, tr in enumerate(trees):
        n = tr.node_count
        base = t * k
        idx = np.arange(n)
        is_leaf = tr.children_left[:n] == -1
        feat[t, :n] = np.where(is_leaf, 0, np.maximum(tr.feature[:n], 0))
        thresh[t, :n] = np.where(is_leaf, 0.0, tr.threshold[:n])
        value[t, :n] = tr.value[:n, 0, 0]
        # children as GLOBAL flat indices; leaves (and padding) self-loop so
        # extra unrolled steps are no-ops once a walker reaches a leaf
        left[t, :n] = base + np.where(is_leaf, idx, tr.children_left[:n])
        right[t, :n] = base + np.where(is_leaf, idx, tr.children_right[:n])
        pad = base + np.arange(n, k)
        left[t, n:] = pad
        right[t, n:] = pad

    # largest float32 <= threshold: exact float32 emulation of sklearn's
    # float32-x-vs-float64-threshold comparison (see module docstring)
    thresh32 = thresh.astype(np.float32)
    over = thresh32.astype(np.float64) > thresh
    thresh32[over] = np.nextafter(thresh32[over], np.float32(-np.inf), dtype=np.float32)

    offsets = (np.arange(n_trees, dtype=np.int64) * k)[None, :]  # [1, T]

    init = [
        numpy_helper.from_array(feat.ravel(), "feat"),
        numpy_helper.from_array(thresh32.ravel(), "thresh"),
        numpy_helper.from_array(left.ravel(), "left"),
        numpy_helper.from_array(right.ravel(), "right"),
        numpy_helper.from_array(value.ravel(), "value"),
        numpy_helper.from_array(offsets, "offsets"),
        numpy_helper.from_array(np.array([0], dtype=np.int64), "zero"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "one"),
        numpy_helper.from_array(np.array([n_trees], dtype=np.int64), "t_dim"),
        numpy_helper.from_array(np.array([1], dtype=np.int64), "axis1"),
    ]

    nodes = [
        # node0 = broadcast tree-root offsets to [N, T]
        helper.make_node("Shape", ["X"], ["x_shape"]),
        helper.make_node("Slice", ["x_shape", "zero", "one", "zero"], ["n_dim"]),
        helper.make_node("Concat", ["n_dim", "t_dim"], ["nt_shape"], axis=0),
        helper.make_node("Expand", ["offsets", "nt_shape"], ["node0"]),
    ]
    node_in = "node0"
    for step in range(depth):
        s = f"_{step}"
        nodes += [
            helper.make_node("Gather", ["feat", node_in], [f"fi{s}"]),
            helper.make_node("GatherElements", ["X", f"fi{s}"], [f"xv{s}"], axis=1),
            helper.make_node("Gather", ["thresh", node_in], [f"th{s}"]),
            helper.make_node("LessOrEqual", [f"xv{s}", f"th{s}"], [f"go_left{s}"]),
            helper.make_node("Gather", ["left", node_in], [f"l{s}"]),
            helper.make_node("Gather", ["right", node_in], [f"r{s}"]),
            helper.make_node("Where", [f"go_left{s}", f"l{s}", f"r{s}"], [f"node{s}"]),
        ]
        node_in = f"node{s}"
    nodes += [
        helper.make_node("Gather", ["value", node_in], ["leaf_vals"]),
        helper.make_node("ReduceMean", ["leaf_vals", "axis1"], ["variable"], keepdims=1),
    ]

    graph = helper.make_graph(
        nodes,
        name,
        [helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, n_features])],
        [helper.make_tensor_value_info("variable", TensorProto.FLOAT, [None, 1])],
        initializer=init,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET)])
    onnx.checker.check_model(model)
    return model
