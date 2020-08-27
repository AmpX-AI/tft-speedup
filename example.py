import math
import os
import random
import time
from copy import deepcopy

import click
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from tf_utils import ColumnTypeInfo, ColumnTypes
from tft_model import TemporalFusionTransformer

physical_devices = tf.config.list_physical_devices("GPU")
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)


class TimeKeeper(tf.keras.callbacks.Callback):
    def __init__(self):
        tf.keras.callbacks.Callback.__init__(self)
        self.logs_history = []

    def on_epoch_end(self, epoch, logs=None):
        """Save custom snapshot of history."""
        self.logs_history.append((time.time(), deepcopy(logs)))


@click.command()
@click.option("--variant", default="base", help="")  # or vectorized
def run_simple_experiment(variant):
    """Base command that allows us to run either setup or both and plot their losses wrt to time."""
    if variant == "all":
        results_v = simple_experiment("vectorized")
        results_b = simple_experiment("base")

        v_x = [point[0] - results_v[0] for point in results_v[1]]
        v_y = [point[1]["val_loss"] for point in results_v[1]]

        b_x = [point[0] - results_b[0] for point in results_b[1]]
        b_y = [point[1]["val_loss"] for point in results_b[1]]

        fig, ax = plt.subplots()
        plt.plot(v_x, v_y, "go-")
        plt.plot(b_x, b_y, "bo-")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Validation loss")
        fig.suptitle("Vectorized vs base version", fontsize=18)
        plt.savefig(f"comparison{str(time.time() / 1000)}.svg")
        plt.close(fig)

        limit = min(v_x[-1], b_x[-1])
        b_x = b_x[: min(len([x for x in b_x if x <= limit]) + 1, len(b_x))]
        b_y = b_y[: len(b_x)]
        v_x = v_x[: min(len([x for x in v_x if x <= limit]) + 1, len(b_x))]
        v_y = v_y[: len(v_x)]

        fig, ax = plt.subplots()
        plt.plot(v_x, v_y, "go-")
        plt.plot(b_x, b_y, "bo-")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Validation loss")
        fig.suptitle("Vectorized vs base version", fontsize=18)
        plt.savefig(f"shorter-comparison{str(time.time() / 1000)}.svg")
        plt.close(fig)

    else:
        simple_experiment(variant)


def simple_experiment(variant):
    """
    On GeForce RTX 2070 with Max-Q Design 8GB, the parameters
    total_data_len = 1000, batch_size = 1 do reflect roughly the maximal size we could fit into GPU memory.

    In a real world case, the (much longer, possibly out-of-core data) would be fed into the model by cut
     parts of the sequence in a generator.
    On the contrary, the base (nonvectorized) version does use a generator, because it would need to be present anyway for
        out-of-core sequences.
    To be fair, the base version does make use of a bigger batch size, since it can fit it into memory.
    (Making the batch sizes equal would make the difference in times even bigger.)

    """
    model_seq_len = 100
    model_future_horizons = 10
    total_data_len = 1000
    variations = 32
    hidden_dependency_len = 10

    SEED = 1331
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    if variant == "vectorized":
        batch_size = 1
        epochs = 13
        # epochs = 2
    else:
        batch_size = 32
        epochs = 4
        # epochs = 2

    def create_data():
        # known_data = np.mod(np.tile(np.arange(0, total_data_len + model_future_horizons), (variations, 1)),
        #                    model_future_horizons)
        known_data = np.mod(
            np.mgrid[0 : total_data_len + model_future_horizons : 1] * np.expand_dims(np.mgrid[0:variations:1].T, -1)
            + np.expand_dims(np.mgrid[0:variations:1].T, -1),
            model_future_horizons,
        )
        # note that the known data are longer `+ model_future_horizons` than all the other,
        # that's because it should contain also future predictions in one sequence
        observed_nontargets_data = np.random.uniform(size=(variations, total_data_len))
        observed_nontargets_data = observed_nontargets_data + np.cos(np.arange(0, total_data_len))
        # hidden rules:
        targets_hidden_dependency = [
            observed_nontargets_data[:, i - hidden_dependency_len : i - model_future_horizons]
            if i > hidden_dependency_len
            else observed_nontargets_data[:, 0:i]
            for i in range(0, known_data.shape[-1])
        ]
        series_hidden_dependency = [
            observed_nontargets_data[:, i - 10 - hidden_dependency_len : i - hidden_dependency_len]
            if i > hidden_dependency_len + 10
            else observed_nontargets_data[:, 0:i]
            for i in range(0, known_data.shape[-1])
        ]
        targets = np.stack(
            [
                np.sum(dependent >= 0.5, axis=-1) % known_data[:, i] + np.sin(np.sum(s_dependency))
                for s_dependency, dependent, i in zip(
                    series_hidden_dependency, targets_hidden_dependency, range(known_data.shape[-1]),
                )
            ],
            axis=-1,
        )
        targets = targets + np.random.uniform(-0.1, 0.1, size=targets.shape)
        # final data:
        targets_format_predicted = [
            targets[:, (i + model_seq_len) : (i + model_seq_len + model_future_horizons)]
            for i in range(total_data_len - model_seq_len + 1)
        ]
        targets_format_predicted = np.stack(targets_format_predicted, axis=1)
        vectorized_input = {
            "known": np.expand_dims(known_data, -1).astype(float),
            "observed": np.stack(
                [observed_nontargets_data, targets[:, : observed_nontargets_data.shape[-1]]], axis=-1,
            ).astype(float),
        }
        vectorized_target_predictions = np.expand_dims(targets_format_predicted, -1).astype(float)

        return vectorized_input, vectorized_target_predictions

    vectorized_input, vectorized_target_predictions = create_data()
    val_vectorized_input, val_vectorized_target_predictions = create_data()

    def np_generator(vectorized_input, vectorized_target_predictions, include_targets=True):
        while True:
            for i in range(total_data_len - model_seq_len + 1):
                for batch in range(int(math.ceil(vectorized_input["known"].shape[0] / batch_size))):
                    window_known = vectorized_input["known"][
                        batch * batch_size : (batch + 1 * batch_size), i : i + model_seq_len, :,
                    ]
                    window_future_known = vectorized_input["known"][
                        batch * batch_size : (batch + 1 * batch_size),
                        i + model_seq_len : i + model_seq_len + model_future_horizons,
                        :,
                    ]
                    window_observed = vectorized_input["observed"][
                        batch * batch_size : (batch + 1 * batch_size), i : i + model_seq_len, :,
                    ]
                    if include_targets:
                        w_target = vectorized_target_predictions[batch * batch_size : (batch + 1 * batch_size), i, ...]
                        yield {
                            "known": window_known,
                            "future-known": window_future_known,
                            "observed": window_observed,
                        }, w_target
                    else:
                        yield {
                            "known": window_known,
                            "future-known": window_future_known,
                            "observed": window_observed,
                        }

    def np_generator_steps(vectorized_input):
        return (total_data_len - model_seq_len + 1) * int(math.ceil(vectorized_input["known"].shape[0] / batch_size))

    def input_types():
        columns_ordering = ["observed", "known", "targets"]

        def get_cti(name_list):
            return ColumnTypeInfo(names=name_list, loc=[columns_ordering.index(name) for name in name_list],)

        return ColumnTypes(
            known_inputs=get_cti(["known"]),
            observed_inputs=get_cti(["observed"]),
            forecast_inputs=get_cti(["targets"]),
            static_inputs=ColumnTypeInfo(),
        )

    tft_model = TemporalFusionTransformer(
        input_shape=[model_seq_len, 2],
        output_shape=[model_future_horizons, 1],
        column_types=input_types(),
        future_size=model_future_horizons,
        num_encoder_steps=model_seq_len,
        dropout_rate=0.1,
        hidden_layer_size=5,
        num_heads=4,
        last_activation="linear",
        static_lookup_size=0,
    )
    if variant == "vectorized":
        model = tft_model.get_model_vectorized(model_capable_vectorize=True, single_sequence=True)
    else:
        model = tft_model.get_model_named_inputs()

    model.compile(optimizer="adam", loss="mse")

    # tf.keras.utils.plot_model(
    #    model, to_file=f"model{variant}.png",
    #    show_shapes=True, show_layer_names=True, rankdir="TB", expand_nested=False, dpi=96,
    # )

    start_time = time.time()

    keeper = TimeKeeper()

    if variant == "vectorized":
        model.fit(
            x=vectorized_input,
            y=vectorized_target_predictions,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(val_vectorized_input, val_vectorized_target_predictions),
            callbacks=[keeper],
        )
    else:
        model.fit(
            np_generator(vectorized_input, vectorized_target_predictions, include_targets=True),
            steps_per_epoch=np_generator_steps(vectorized_input),
            validation_data=np_generator(
                val_vectorized_input, val_vectorized_target_predictions, include_targets=True,
            ),
            validation_steps=np_generator_steps(val_vectorized_input),
            epochs=epochs,
            verbose=1,
            callbacks=[keeper],
        )
    stop_time = time.time()
    print(f"Run took {stop_time - start_time} s")
    return start_time, keeper.logs_history, stop_time


if __name__ == "__main__":
    run_simple_experiment()
