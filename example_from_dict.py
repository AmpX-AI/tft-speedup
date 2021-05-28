import os
import random
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf

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


embedd_by_divisors = [1000.0, 100.0, 10.0, 1.0]
embedd_by_functions = [lambda x: x, lambda x: np.sin(x * np.pi), lambda x: np.cos(x * np.pi)]


def embedd_array(x, functions=embedd_by_functions, mults=embedd_by_divisors):
    all = []
    for mult in mults:
        all.extend([function(x * mult) for function in functions])
    return np.concatenate(all, axis=-1)


"""
While the paper schematics (https://images.deepai.org/converted-papers/1912.09363/Schematic.png) shows
"known, observed and static" features/inputs,
the TFT implementation (https://github.com/google-research/google-research/blob/master/tft/libs/tft_model.py)
uses the notations:
- known inputs 
  - future known inputs  (difference being just where we split each window in a datapoint to be trained)
  - past known inputs
- unknown inputs
  = not observed and not known inputs 
- observed inputs
  = are targets (also assumed to be only realvalued, since the targets of tft are also realvalued)
  (can be seen at tft/data_formatters/base.py, 'input_obs_loc')

Observed and Unknown are, nevertheless, handled in the same way and just concatenated/stacked in the end.

Notes:
- Categorical variables are -in the google code- embedded using:
    tf.keras.layers.InputLayer([time_steps]),
              tf.keras.layers.Embedding(
    (Which applies even for static inputs)


Our implementation allows only for the notation of unknown, observed, known and static,
 while handling unknown & observed in the same way.
Our implementation assumes, that all embeddings and value type conversions are done beforehand and so ignores all the 
TFT embeddings. 
"""


def run_simple_experiment():
    model_seq_len = 100
    model_future_horizons = 10

    SEED = 1331
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)

    batch_size = 1
    epochs = 13

    columns_ordering = ["day_of_week",
                        "hour",
                        "weather_parameters_1",
                        "weather_parameters_2",
                        "target_variable",
                        ]
    target_column_name = "target_variable"

    def create_data():
        """Example that creates data from a dictionary of columns.
        Could/should be replaced with pandas reading files from csv.

        Note that the columns have different lengths
        (thats why I have omitted pandas for now, as that is a preprocessing).

        TODOS:
            - The data are being fed as in the basic Keras MNIST examples - using just numpy arrays and not generators
              - The generators need to cut the sequences to care for the GPU memory (and decide about batch size)
            - If the data consist of different series for different people, we need to cut them (in the generators)
                to never include/feed data with two person_ids at once to the model,
                as no preprocessing from TFT original is used here.
            - Anything written below as "TODO"
        """
        base_len = 200
        # this dictionary supplements a source dataframe queried for the data.
        data_columns = {
            # observed variables are of given base len
            "weather_parameters_1": np.zeros(shape=(base_len, 1)),
            "weather_parameters_2": np.zeros(shape=(base_len, 1)),
            # known variables are known also in the future (after the observed variables end)
            "day_of_week": np.zeros(base_len + model_future_horizons),
            "hour": np.zeros(base_len + model_future_horizons),
            # target variables are also known in the future
            "target_variable": np.zeros(base_len + model_future_horizons),
            # static variable is just one categorical for example
            "person_id": 0,
        }

        processed_data_per_column = {}
        for column in columns_ordering:
            if data_columns[column].ndim > 1:
                processed_data_per_column[column] = data_columns[column]
            else:
                processed_data_per_column[column] = np.expand_dims(data_columns[column], -1)
                # if we have a less dimensional input, we need to add a feature space dimension

            # TODO: plus any normalization that should take place in any data preprocessing for Neural Nets

        # Using embeddings from attention is all you need:
        person_processing = data_columns["person_id"]
        processed_data_per_column["person_id"] = embedd_array(np.array([person_processing]))

        # now lets group the data into dicts as inputs to TFT:
        x = {
            "known": np.concatenate([processed_data_per_column["day_of_week"],
                                     processed_data_per_column["hour"],
                                     ], axis=-1),
            "observed": np.concatenate([processed_data_per_column["weather_parameters_1"],
                                        processed_data_per_column["weather_parameters_2"],
                                        processed_data_per_column[target_column_name]
                                        [:len(processed_data_per_column["weather_parameters_1"])],
                                        # ^ note that the observed target cannot be observed in the future and thats why we cut it
                                        ], axis=-1),
            "static": processed_data_per_column["person_id"]

        }
        # add batch dimension (=1):
        x = {item: np.expand_dims(x[item], 0) for item in x}

        # create target varaible, i.e at each timestep, produce a prediction of model_future_horizons
        y_targets = []
        for y_ind in range(model_seq_len,
                           len(processed_data_per_column[target_column_name]) - model_future_horizons + 1):
            y_targets.append(processed_data_per_column[target_column_name][y_ind:y_ind + model_future_horizons])

        y = np.stack(y_targets, 0)
        y = np.expand_dims(y, 0)  # batch dimension fortargets

        return x, y

    vectorized_input, vectorized_target_predictions = create_data()
    val_vectorized_input, val_vectorized_target_predictions = create_data()

    tft_model = TemporalFusionTransformer(
        input_shape=[model_seq_len, 2],
        output_shape=[model_future_horizons, 1],
        n_known=2,  # see the generator
        n_observed=3,  # see the generator
        future_size=model_future_horizons,
        num_encoder_steps=model_seq_len,
        dropout_rate=0.1,
        hidden_layer_size=5,
        num_heads=4,
        last_activation="linear",
        static_lookup_size=12,  # see the generator
    )
    model = tft_model.get_model_vectorized(model_capable_vectorize=True, single_sequence=True)
    model.compile(optimizer="adam", loss="mse")
    try_y = model(vectorized_input)  # just a check to see our model works on the inputs
    assert try_y is not None
    start_time = time.time()
    keeper = TimeKeeper()
    model.fit(
        x=vectorized_input,
        y=vectorized_target_predictions,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(val_vectorized_input, val_vectorized_target_predictions),
        callbacks=[keeper],
    )
    stop_time = time.time()
    print(f"Run took {stop_time - start_time} s")
    return start_time, keeper.logs_history, stop_time


if __name__ == "__main__":
    run_simple_experiment()
