from __future__ import annotations

import abc
import enum
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Union, Callable

import numpy as np
import tensorflow as tf
from keras.engine.keras_tensor import KerasTensor


def s_len(xslice: slice) -> int:
    """Get length of slice."""
    return xslice.stop - xslice.start

def flatten_deep(xlist: list):
    """Recursively flattens (in depth first search manner) a structure containing lists and tuples."""
    ret = []
    for item in xlist:
        if isinstance(item, list) or isinstance(item, tuple):
            ret.extend(flatten_deep(item))
        else:
            ret.append(item)
    return ret

def expand_to_match_first_dims(indices, orig_seq, batch_dims):
    """Inserts batch_dims of orig_seq.shape[:batch_dims] dimensions to indices to make thise first dimensions match.

    The rest of input indices remains unchanged (but duplicated along the new dimensions).
    """
    for i in range(batch_dims):
        indices = tf.expand_dims(indices, 0)
    tile_match = tf.concat([tf.shape(orig_seq)[:batch_dims], [1]], -1)
    return tf.tile(indices, tile_match)


def timedistributed_over_more_batch_dimensions(
    op: Callable, batch_dims: int, seq: Union[KerasTensor, tf.Tensor], **kwargs
):
    """A variant of Keras TimeDistributed wrapper layer to use more batch dimensions.

    Args:
        op: Operation to be called with seq and kwargs.
        batch_dims: Number of batch dimensions to skip (vectorize over).
        seq: Sequence or multidimensional data input.

    Returns:
        The result of the operation with the batch dimensions unchanged.
    """
    if batch_dims <= 1:
        return op(seq, **kwargs)
    else:
        seq_squashed, batch_shape_orig = squash_batch_dimensions(seq, batch_dims)
        results_squashed = op(seq_squashed, **kwargs)
        if isinstance(results_squashed, list) or isinstance(results_squashed, tuple):
            return [unsquash_batch_dimensions(results, batch_shape_orig) for results in results_squashed]
        else:
            return unsquash_batch_dimensions(results_squashed, batch_shape_orig)


def timedistributed_of_lstm_state(op, batch_dims, seq, initial_state, **kwargs):
    """A variant of Keras TimeDistributed wrapper layer to use more batch dimensions with initial state.

    Args:
        op: Operation to be called with seq and kwargs
        batch_dims: Number of batch dimensions to skip (vectorize over)
        seq: Sequence or multidimensional data input
        initial_state: Initial states to be passed to op.

    Returns:
        The result of the operation with the batch dimensions unchanged. And returned unsquashed lstm states.
    """
    if batch_dims <= 1:
        return op(seq, **kwargs)
    else:
        seq_squashed, batch_shape_orig = squash_batch_dimensions(seq, batch_dims)
        multiplies = tf.math.reduce_prod(batch_shape_orig[1:])
        aligned_initial_state = [repeat_multiply_batch_dimension(tnsr, multiplies) for tnsr in initial_state]
        results_squashed = op(seq_squashed, initial_state=aligned_initial_state, **kwargs)
        return (
            unsquash_batch_dimensions(results_squashed[0], batch_shape_orig),
            results_squashed[1],
            results_squashed[2],
        )


def get_shape_fixed_if_possible(tensor):
    """A variant of tf.shape(tensor) but returning a list and leaving a fixed dimension whenever it can."""
    seq_tf_shape = tf.shape(tensor)
    return_shape = [0] * len(tensor.shape)

    # "call return_shape = tf.shape(seq)[:batch_dims], but leave fixed whenever we can:"
    for dim in range(len(tensor.shape)):
        if tensor.shape[dim] is None:
            return_shape[dim] = seq_tf_shape[dim]
        else:
            return_shape[dim] = tensor.shape[dim]
    return return_shape


def squash_batch_dimensions(seq, batch_dims):
    """Squashes first (batch_dims) dimensions into 1 batch dimension."""
    batch_shape_orig = tf.shape(seq)[:batch_dims]
    retain_shape = tf.shape(seq)[batch_dims:]
    # new_shape = tf.concat([[-1], retain_shape], axis=-1)
    # in tf 2.4 - the above should work with -1 but is not. (Report error to tensorflow?)
    new_shape = tf.concat([[tf.math.reduce_prod(batch_shape_orig)], retain_shape], axis=-1)
    seq_squashed = tf.reshape(seq, new_shape)
    return seq_squashed, batch_shape_orig


def repeat_multiply_batch_dimension(static_variable, dim_multiply):
    """Multiplies batch dimension for a static variable to match other squashed input."""
    return tf.repeat(static_variable, dim_multiply, axis=0)


def unsquash_batch_dimensions(seq: tf.Tensor, batch_shape_orig: Union[list, tf.Tensor]):
    """Restores additional dimensions previously squashed to the first batch dimension"""
    batch_dims = 1
    if isinstance(batch_shape_orig, list):
        retain_shape = get_shape_fixed_if_possible(seq)[batch_dims:]
        new_shape = tf.stack(flatten_deep([batch_shape_orig, retain_shape]), axis=-1)
    else:
        retain_shape = tf.shape(seq)[batch_dims:]
        new_shape = tf.concat([batch_shape_orig, retain_shape], axis=-1)
    seq = tf.reshape(seq, new_shape)
    return seq


def windowing_mechanism(seq_data, batch_dims, window_len):
    """Reformats data in timesequence dimension into two dimensions of [number of windows, window length] by windowing.

    :param seq_data: Original data with time dimension at seq_data.shape[batch_dims] position
        and features at seq_data.shape[batch_dims:]
    :param batch_dims: Defines the number of dimensions to regard as batch at the start. Expects non-tensor.
    :param window_len: Window length to produce. Expects non-tensor (?)
    :return: windowed original sequence of shape:
        list(data.shape[:batch_dims])
        + [data.shape[batch_dims] - window_len + 1, window_len]
        + list(data.shape[batch_dims + 1 :])
    """

    rep = (
        get_shape_fixed_if_possible(seq_data)[batch_dims] - window_len + 1
    )  # when windowing, we will repeat that many times

    # as is first nonbatch dimension - window len + 1
    i = tf.range(0, window_len * rep, delta=1, dtype=None, name="range")  # now we will produce indexes for gather
    indices_orig = tf.math.floormod(i, window_len) + tf.math.floordiv(i, window_len)
    # in this manner: [[ 1,  2,  3,  4,  5,  ..  2,  3,  4,  5,  6,  ..  3,  4,  5,  6,  7, .. ]]

    # expand indices for aligning with batch dimension to use tf.gather on batched data:
    indices = expand_to_match_first_dims(indices_orig, seq_data, batch_dims)
    # should match all the batch dims before!

    result = tf.gather(seq_data, indices, batch_dims=batch_dims,)
    # now reshape it in the form of the matrix we want:
    new_shape = tf.concat(
        [tf.shape(seq_data)[:batch_dims], [rep, window_len], tf.shape(seq_data)[batch_dims + 1 :]], axis=-1
    )
    windowed = tf.reshape(result, new_shape)
    return windowed
    # ... now regard the first (batch_dims+1) dimensions as batches for any model to apply


def check_activation_len(layers, activations):
    if isinstance(activations, (str, type(None))):
        activations = [activations] * len(layers)
    assert len(layers) == len(activations)
    return activations


class QuantileLoss(tf.keras.losses.Loss):
    def __init__(self, quantiles: List = [0.1, 0.5, 0.9], targets_count: int = 1):
        self.quantiles = quantiles
        self.targets_count = (
            targets_count  # in case we have multiple targets => output dim[-1] = output_size * n_quantiles
        )
        self._q_multipliers = np.array(self.quantiles * self.targets_count, dtype=float)
        super().__init__()

    def call(self, y_true, y_pred):
        """Computes the quantile loss from targets and sources in the tf graph.

        Assumes the same format as is introduced in `IndexedFeed.repeat_targets_for_quantiles`.
        Example:
            (two targets, three quantiles:)
            t-p = error * self._q_multipliers
            1 1           q1
            1 1           q2
            1 1           q3
            2 2           q1
            2 2           q2
            2 2           q3
        """
        error = tf.subtract(y_true[..., :], y_pred[..., :])
        same_type_q = tf.cast(tf.constant(self._q_multipliers, dtype=tf.float64), error.dtype)
        return tf.reduce_mean(tf.maximum(same_type_q * error, (same_type_q - 1.0) * error))


class TFTInputNames(str, enum.Enum):
    FUTURE_KNOWN_INPUT = "future-known"  # Only for non sequential TFT implementations
    STATIC_INPUT = "static"
    OBSERVED_INPUT = "observed"
    KNOWN_INPUT = "known"
