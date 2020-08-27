import copy
import inspect
import os
import tempfile

import numpy as np
import tensorflow as tf

from subprocessify import subprocessify
from tft_model import TemporalFusionTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force the tests to run on CPU (to not clutter GPU memory)

try:  # old tf version
    from tensorflow.python.keras.engine.network import get_network_config
except Exception:  # updated version - for future to know where this function lies
    from tensorflow.python.keras.engine.functional import get_network_config


def try_deepcopy(x):
    try:
        return [True, copy.deepcopy(x)]
    except Exception:
        return [False, x]


def assert_model_saves(model):
    xconfig = get_network_config(model)
    dpcp = [try_deepcopy(layer) for layer in xconfig["layers"]]

    problems = [item for item in dpcp if not item[0]]
    if len(problems) > 0:
        problems_descriptions = [
            inspect.getsourcelines(model.get_layer(problem[1]["name"]).function) for problem in problems
        ]
        problems_descriptions = ["{}:{}".format(item[1], item[0]) for item in problems_descriptions]
        assert len(problems) <= 0, "The following layers could not be saved by tf standard routines: {}".format(
            "\n ".join(problems_descriptions)
        )

    xconfig = get_network_config(model)
    copied = copy.deepcopy(xconfig)
    assert copied is not None


Lambda = tf.keras.layers.Lambda


@subprocessify(True)
def test_tft_model_saving():
    """WIP test, currently checks if there are any problems in TFT model during saving.

    Should also test loading, but that seems to be broken on TF part (see below).
    """
    model_params = {
        "dropout_rate": 0.1,
        "hidden_layer_size": 5,
        "num_heads": 4,
        "num_encoder_steps": 7 * 24,
        "input_shape": [8 * 24, 5],
    }

    column_params = {
        "output_dim": 1,
        "input_obs_loc": [0],
        "static_input_loc": [],
        "known_regular_inputs": [1, 2, 3],
    }

    params = {**model_params, **column_params}
    tft = TemporalFusionTransformer(**params)
    model = tft.get_model()

    assert_model_saves(model)

    # try that it can load the weights at least:
    tmpdir = tempfile.TemporaryDirectory()
    model.save(tmpdir.name + "/model_saved.h5", save_format="h5")
    model.load_weights(tmpdir.name + "/model_saved.h5")


@subprocessify(True)
def test_tft_model_orig_saving(input_types_statics, try_eval=False):
    """Test that checks if there are any problems in TFT model during saving."""
    model_params = {
        "dropout_rate": 0.1,
        "hidden_layer_size": 5,
        "num_heads": 4,
        "num_encoder_steps": 7 * 24,
        "input_shape": [8 * 24, 6],
    }

    column_params = {
        "output_dim": 1,
        "input_obs_loc": [0],
        "known_regular_inputs": [1, 2, 3],
        "static_lookup_size": 0,
    }

    params = {**model_params, **column_params}
    tft = TemporalFusionTransformer(output_shape=[24, 1], column_types=input_types_statics, **params)
    model = tft.get_model()

    zero_input = {}
    for input in model.inputs:
        zero_input[input.name.replace(":0", "")] = np.zeros(
            [size if size is not None else 1 for size in input.shape], dtype=input.dtype.as_numpy_dtype,
        )
    if try_eval:
        evaluated_model = model(zero_input)
        assert evaluated_model is not None

    assert_model_saves(model)

    # try that it can load the weights:
    tmpdir = tempfile.TemporaryDirectory()
    model.save(tmpdir.name + "/model_saved.h5", save_format="h5")
    model.load_weights(tmpdir.name + "/model_saved.h5")


@subprocessify(True)
def test_tft_model_named_inputs_saving(input_types, try_eval=False):
    """Test that checks if there are any problems in TFT model during saving."""
    model_params = {
        "dropout_rate": 0.1,
        "hidden_layer_size": 5,
        "num_heads": 4,
        "num_encoder_steps": 7 * 24,
        "input_shape": [8 * 24, 5],
    }

    column_params = {
        "output_dim": 1,
        "input_obs_loc": [0],
        "static_input_loc": [],
        "known_regular_inputs": [1, 2, 3],
        "static_lookup_size": 4,
    }

    params = {**model_params, **column_params}
    tft = TemporalFusionTransformer(output_shape=[24, 1], column_types=input_types, **params)
    model = tft.get_model_named_inputs()

    zero_input = {}
    for input in model.inputs:
        zero_input[input.name.replace(":0", "")] = np.zeros(
            [size if size is not None else 1 for size in input.shape], dtype=input.dtype.as_numpy_dtype,
        )
    if try_eval:
        evaluated_model = model(zero_input)
        assert evaluated_model is not None

    assert_model_saves(model)

    # try that it can load the weights:
    tmpdir = tempfile.TemporaryDirectory()
    model.save(tmpdir.name + "/model_saved.h5", save_format="h5")
    model.load_weights(tmpdir.name + "/model_saved.h5")


@subprocessify(True)
def test_tft_model_vectorized_saving(input_types, try_eval=True):
    """Checks if there are any problems in TFT model during saving.

    Also checks that the vectorized model works on sequential inputs.
    """
    model_params = {
        "dropout_rate": 0.1,
        "hidden_layer_size": 5,
        "num_heads": 4,
        "num_encoder_steps": 7 * 24,
        "input_shape": [8 * 24, 5],
    }

    column_params = {
        "output_dim": 1,
        "input_obs_loc": [0],
        "static_input_loc": [],
        "known_regular_inputs": [1, 2, 3],
        "static_lookup_size": 4,
    }

    params = {**model_params, **column_params}
    tft = TemporalFusionTransformer(output_shape=[24, 1], column_types=input_types, **params)
    model = tft.get_model_vectorized(single_sequence=False)

    def try_inputs(model, hist_size, future_size, added_seq_len):
        zero_input = {}
        for input in model.inputs:
            name = input.name.replace("_0", "").replace(":0", "")
            if name == "future-known":
                zero_input[name] = np.zeros(
                    [1, future_size + added_seq_len, input.shape[-1]], dtype=input.dtype.as_numpy_dtype,
                )
            elif name == "static":
                zero_input[name] = np.zeros([1, input.shape[-1]], dtype=input.dtype.as_numpy_dtype)
            else:
                zero_input[name] = np.zeros(
                    [1, hist_size + added_seq_len, input.shape[-1]], dtype=input.dtype.as_numpy_dtype,
                )
        return zero_input

    if try_eval:
        zero_input = try_inputs(model, tft.num_encoder_steps, tft.get_future_fork_size(), 0)
        evaluated_model = model(zero_input)
        assert evaluated_model.shape == (
            1,
            1,
            tft.get_future_fork_size(),
            tft.output_dim,
        ), "the vectorized model should work on, at least, the same inputs as the original model"

        try_rel_seq_len = 10
        zero_input = try_inputs(model, tft.num_encoder_steps, tft.get_future_fork_size(), try_rel_seq_len)
        evaluated_model = model(zero_input)
        assert evaluated_model.shape == (
            1,
            try_rel_seq_len + 1,
            tft.get_future_fork_size(),
            tft.output_dim,
        ), "the vectorized model should return a sequence of a provided length ({})".format(try_rel_seq_len + 1)

    assert_model_saves(model)

    # try that it can load the weights:
    tmpdir = tempfile.TemporaryDirectory()
    model.save(tmpdir.name + "/model_saved.h5", save_format="h5")
    model.load_weights(tmpdir.name + "/model_saved.h5")
    return True


@subprocessify(True)
def test_tft_single_sequence(input_types, try_eval=True):
    """Checks if there are any problems in TFT model during saving.

    Also checks that the vectorized model works on sequential inputs.
    """
    model_params = {
        "dropout_rate": 0.1,
        "hidden_layer_size": 5,
        "num_heads": 4,
        "num_encoder_steps": 96,
        "future_size": 12,
        "input_shape": [4 * 24, 5],
    }

    column_params = {
        "output_dim": 1,
        "input_obs_loc": [0],
        "static_input_loc": [],
        "known_regular_inputs": [1, 2, 3],
        "static_lookup_size": 4,
    }
    output_shape = [12, 1]

    params = {**model_params, **column_params}
    tft = TemporalFusionTransformer(output_shape=output_shape, column_types=input_types, **params)
    model = tft.get_model_vectorized(single_sequence=True)

    def try_inputs(model, hist_size, future_size, added_seq_len):
        zero_input = {}
        for input in model.inputs:
            name = input.name.replace("_0", "").replace(":0", "")
            if name == "known":
                zero_input[name] = np.zeros(
                    [1, hist_size + future_size + added_seq_len, input.shape[-1]], dtype=input.dtype.as_numpy_dtype,
                )
            elif name == "static":
                zero_input[name] = np.zeros([1, input.shape[-1]], dtype=input.dtype.as_numpy_dtype)
            else:
                zero_input[name] = np.zeros(
                    [1, hist_size + added_seq_len, input.shape[-1]], dtype=input.dtype.as_numpy_dtype,
                )
        return zero_input

    if try_eval:
        zero_input = try_inputs(model, tft.num_encoder_steps, tft.get_future_fork_size(), 0)
        evaluated_model = model(zero_input)
        assert evaluated_model.shape == (
            1,
            1,
            tft.get_future_fork_size(),
            tft.output_dim,
        ), "the vectorized model should work on, at least, the same inputs as the original model"

        try_rel_seq_len = 291
        zero_input = try_inputs(model, tft.num_encoder_steps, tft.get_future_fork_size(), try_rel_seq_len)
        evaluated_model = model(zero_input)
        assert evaluated_model.shape == (
            1,
            try_rel_seq_len + 1,
            tft.get_future_fork_size(),
            tft.output_dim,
        ), "the vectorized model should return a sequence of a provided length ({})".format(try_rel_seq_len + 1)

    assert_model_saves(model)

    # try that it can load the weights:
    tmpdir = tempfile.TemporaryDirectory()
    model.save(tmpdir.name + "/model_saved.h5", save_format="h5")
    model.load_weights(tmpdir.name + "/model_saved.h5")
