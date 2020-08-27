from __future__ import annotations

import logging
from typing import Dict, Tuple

import tensorflow as tf

from tf_utils import (
    get_shape_fixed_if_possible,
    repeat_multiply_batch_dimension,
    squash_batch_dimensions,
    timedistributed_of_lstm_state,
    timedistributed_over_more_batch_dimensions,
    unsquash_batch_dimensions,
    windowing_mechanism,
    ColumnTypes,
)

logger = logging.getLogger(__name__)

Layer = tf.keras.layers.Layer
Lambda = tf.keras.layers.Lambda
K = tf.keras.backend


def tf_stack(x, axis=0):
    if not isinstance(x, list):
        # when loading, tensorflow sometimes forgets...
        x = [x]
    return K.stack(x, axis=axis)


# Attention Components.
def get_decoder_mask(self_attn_inputs):
    """Returns causal mask to apply for self-attention layer.

    Args:
        self_attn_inputs: Inputs to self attention layer to determine mask shape
    """
    len_s = tf.shape(self_attn_inputs)[-2]
    bs = tf.shape(self_attn_inputs)[:-2]
    mask = tf.cumsum(tf.eye(len_s, batch_shape=bs), -2)
    return mask


def add_and_norm(x_list):
    """Applies skip connection followed by layer normalisation.

    Args:
        x_list: List of inputs to sum for skip connection

    Returns:
        Tensor output from layer.
    """
    tmp = tf.keras.layers.Add()(x_list)
    tmp = tf.keras.layers.LayerNormalization()(tmp)
    return tmp


def linear_layer(size, activation=None, use_time_distributed=False, use_bias=True):
    """Returns simple Keras linear layer.

    Args:
        size: Output size
        activation: Activation function to apply if required
        use_time_distributed: Whether to apply layer across time
        use_bias: Whether bias should be included in layer
    """
    linear = tf.keras.layers.Dense(size, activation=activation, use_bias=use_bias)
    if use_time_distributed:
        linear = tf.keras.layers.TimeDistributed(linear)
    return linear


def apply_gating_layer(
    x, hidden_layer_size: int, dropout_rate: float = None, use_time_distributed: bool = True, activation=None,
):
    """Applies a Gated Linear Unit (GLU) to an input.

    Args:
        x: Input to gating layer
        hidden_layer_size: Dimension of GLU
        dropout_rate: Dropout rate to apply if any
        use_time_distributed: Whether to apply across time
        activation: Activation function to apply to the linear feature transform if necessary

    Returns:
        Tuple of tensors for: (GLU output, gate)
    """

    if dropout_rate is not None:
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    if use_time_distributed:
        activation_layer = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(hidden_layer_size, activation=activation)
        )(x)
        gated_layer = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(hidden_layer_size, activation="sigmoid"))(x)
    else:
        activation_layer = tf.keras.layers.Dense(hidden_layer_size, activation=activation)(x)
        gated_layer = tf.keras.layers.Dense(hidden_layer_size, activation="sigmoid")(x)

    return tf.keras.layers.multiply([activation_layer, gated_layer]), gated_layer


def gated_residual_network(
    x,
    hidden_layer_size: int,
    output_size: int = None,
    dropout_rate: float = None,
    use_time_distributed: bool = True,
    additional_context=None,
    return_gate: bool = False,
):
    """Applies the gated residual network (GRN) as defined in paper.

    Args:
        x: Network inputs
        hidden_layer_size: Internal state size
        output_size: Size of output layer
        dropout_rate: Dropout rate if dropout is applied
        use_time_distributed: Whether to apply network across time dimension
        additional_context: Additional context vector to use if relevant
        return_gate: Whether to return GLU gate for diagnostic purposes

    Returns:
        Tuple of tensors for: (GRN output, GLU gate)
    """

    # Setup skip connection
    if output_size is None:
        output_size = hidden_layer_size
        skip = x
    else:
        linear = tf.keras.layers.Dense(output_size)
        if use_time_distributed:
            linear = tf.keras.layers.TimeDistributed(linear)
        skip = linear(x)

    # Apply feedforward network
    hidden = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed)(x)
    if additional_context is not None:
        hidden = hidden + linear_layer(
            hidden_layer_size, activation=None, use_time_distributed=use_time_distributed, use_bias=False,
        )(additional_context)
    hidden = tf.keras.layers.Activation("elu")(hidden)
    hidden = linear_layer(hidden_layer_size, activation=None, use_time_distributed=use_time_distributed)(hidden)

    gating_layer, gate = apply_gating_layer(
        hidden, output_size, dropout_rate=dropout_rate, use_time_distributed=use_time_distributed, activation=None,
    )

    if return_gate:
        return add_and_norm([skip, gating_layer]), gate
    else:
        return add_and_norm([skip, gating_layer])


def tempering_batchdot(input_list):
    d, k = input_list
    temper = tf.sqrt(tf.cast(k.shape[-1], dtype="float32"))
    return K.batch_dot(d, k, axes=[2, 2]) / temper


class ScaledDotProductAttention(tf.keras.layers.Layer):
    """Defines scaled dot product attention layer.

    Attributes:
        dropout: Dropout rate to use
        activation: Normalisation function for scaled dot product attention (e.g. softmax by default)
    """

    def __init__(self, attn_dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.dropout = tf.keras.layers.Dropout(attn_dropout)
        self.activation = tf.keras.layers.Activation("softmax")

    def __call__(self, q, k, v, mask):
        """Applies scaled dot product attention.

        Args:
            q: Queries
            k: Keys
            v: Values
            mask: Masking if required -- sets softmax to very large value

        Returns:
            Tuple of (layer outputs, attention weights)
        """
        attn = Lambda(tempering_batchdot)([q, k])  # shape=(batch, q, k)
        if mask is not None:
            mmask = Lambda(lambda x: (-1e9) * (1.0 - tf.cast(x, "float32")))(mask)  # setting to infinity
            attn = tf.keras.layers.add([attn, mmask])
        attn = self.activation(attn)
        attn = self.dropout(attn)
        output = Lambda(lambda x: K.batch_dot(x[0], x[1]))([attn, v])
        return output, attn


class InterpretableMultiHeadAttention(tf.keras.layers.Layer):
    """Defines interpretable multi-head attention layer.

    Attributes:
        n_head: Number of heads
        d_k: Key/query dimensionality per head
        d_v: Value dimensionality
        dropout: Dropout rate to apply
        qs_layers: List of queries across heads
        ks_layers: List of keys across heads
        vs_layers: List of values across heads
        attention: Scaled dot product attention layer
        w_o: Output weight matrix to project internal state to the original TFT state size
    """

    def __init__(self, n_head: int, d_model: int, dropout: float, **kwargs):
        """Initialises layer.

        Args:
            n_head: Number of heads
            d_model: TFT state dimensionality
            dropout: Dropout discard rate
        """

        super().__init__(**kwargs)
        self.n_head = n_head
        self.d_k = self.d_v = d_k = d_v = d_model // n_head
        self.dropout = dropout

        self.qs_layers = []
        self.ks_layers = []
        self.vs_layers = []

        # Use same value layer to facilitate interp
        vs_layer = tf.keras.layers.Dense(d_v, use_bias=False)

        for _ in range(n_head):
            self.qs_layers.append(tf.keras.layers.Dense(d_k, use_bias=False))
            self.ks_layers.append(tf.keras.layers.Dense(d_k, use_bias=False))
            self.vs_layers.append(vs_layer)  # use same vs_layer

        self.attention = ScaledDotProductAttention()
        self.w_o = tf.keras.layers.Dense(d_model, use_bias=False)

    def __call__(self, q, k, v, mask=None):
        """Applies interpretable multihead attention.

        Using T to denote the number of time steps fed into the transformer.

        Args:
            q: Query tensor of shape=(?, T, d_model)
            k: Key of shape=(?, T, d_model)
            v: Values of shape=(?, T, d_model)
            mask: Masking if required with shape=(?, T, T)

        Returns:
            Tuple of (layer outputs, attention weights)
        """
        n_head = self.n_head

        heads = []
        attns = []
        for i in range(n_head):
            qs = self.qs_layers[i](q)
            ks = self.ks_layers[i](k)
            vs = self.vs_layers[i](v)
            head, attn = self.attention(qs, ks, vs, mask)

            head_dropout = tf.keras.layers.Dropout(self.dropout)(head)
            heads.append(head_dropout)
            attns.append(attn)
        head = Lambda(tf_stack)(heads) if n_head > 1 else heads[0]
        attn = Lambda(tf_stack)(attns)

        outputs = Lambda(K.mean, arguments={"axis": 0})(head) if n_head > 1 else head
        outputs = self.w_o(outputs)
        outputs = tf.keras.layers.Dropout(self.dropout)(outputs)  # output dropout

        return outputs, attn


class TemporalFusionTransformer:
    """Defines Temporal Fusion Transformer.

    Attributes:
        name: Name of model
        input_shape: Total time steps (i.e. Width of Temporal fusion decoder N) x total number of inputs
        output_dim: Total number of outputs
        hidden_layer_size: Internal state size of TFT
        dropout_rate: Dropout discard rate
        num_encoder_steps: Size of LSTM encoder -- i.e. number of past time steps before forecast date to use
        num_heads: Number of heads for interpretable multi-head attention
    """

    def __init__(
        self,
        hidden_layer_size: int,
        dropout_rate: float,
        num_heads: int,
        input_shape: list[int],
        output_shape: list[int] = None,
        column_types: ColumnTypes = None,
        static_lookup_size: int = 0,
        last_activation: str = None,
        **kwargs
    ) -> None:
        """Builds TFT from parameters.

        Args:
            raw_params: Parameters to define TFT
        """

        self.name = self.__class__.__name__

        self.input_shape = input_shape

        if column_types:
            self.output_dim = output_shape[-1]
            if "num_encoder_steps" in kwargs:
                self.num_encoder_steps = kwargs["num_encoder_steps"]
            else:
                self.num_encoder_steps = input_shape[0] - output_shape[0]  # nongenerator window approach
            # ^ num_encoder_steps should be generally equal to seq_len since we use forking sequences training scheme
            if "future_size" in kwargs:
                self.future_size = kwargs["future_size"]
            else:
                self.future_size = self.input_shape[0] - self.num_encoder_steps

            (input_obs_loc, static_input_loc, known_regular_input_idx, knowns, unknowns,) = column_types.get_input_loc()
            self.input_obs_loc = input_obs_loc
            self.static_input_loc = static_input_loc
            self.known_regular_input_idx = known_regular_input_idx
            self.knowns = knowns
            self.unknowns = unknowns
        else:
            # parameters for both get model variants:
            self.output_dim = kwargs["output_dim"]
            self.input_obs_loc = kwargs["input_obs_loc"]
            self.static_input_loc = kwargs["static_input_loc"]
            self.known_regular_input_idx = kwargs["known_regular_inputs"]
            self.num_encoder_steps = kwargs["num_encoder_steps"]

        self.static_lookup_size = static_lookup_size
        # Network params
        self.hidden_layer_size = hidden_layer_size
        self.dropout_rate = dropout_rate

        self.num_heads = num_heads

        self.last_activation = last_activation

        # Extra components to store Tensorflow nodes for attention computations
        self._attention_components = None

    def get_future_fork_size(self):
        """Returns the length of future "fork" the model does expect."""
        return self.future_size

    def get_tft_embeddings(self, all_inputs):
        """Transforms raw inputs to embeddings.

        Applies linear transformation onto continuous variables.

        Args:
            all_inputs: Inputs to transform

        Returns:
            Tensors for transformed inputs.
        """

        # Sanity checks
        for i in self.known_regular_input_idx:
            if i in self.input_obs_loc:
                raise ValueError("Observation cannot be known a priori!")
        for i in self.input_obs_loc:
            if i in self.static_input_loc:
                raise ValueError("Observation cannot be static!")

        # Static inputs
        if self.static_input_loc:
            static_inputs = [
                tf.keras.layers.Dense(self.hidden_layer_size)(all_inputs[:, 0, i : i + 1])
                for i in range(self.input_shape[-1])
                if i in self.static_input_loc
            ]
            static_inputs = Lambda(tf_stack, arguments={"axis": 1})(static_inputs)

        else:
            static_inputs = None

        def convert_real_to_embedding(x):
            """Applies linear transformation for time-varying inputs."""
            return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.hidden_layer_size))(x)

        # Targets
        obs_inputs = Lambda(tf_stack, arguments={"axis": -1})(
            [convert_real_to_embedding(all_inputs[..., i : i + 1]) for i in self.input_obs_loc]
        )

        unknown_inputs = []
        for i in range(all_inputs.shape[-1]):
            if i not in self.known_regular_input_idx and i not in self.input_obs_loc:
                e = convert_real_to_embedding(all_inputs[..., i : i + 1])
                unknown_inputs.append(e)

        if unknown_inputs:
            unknown_inputs = Lambda(tf_stack, arguments={"axis": -1})(unknown_inputs)
        else:
            unknown_inputs = None

        # A priori known inputs
        known_regular_inputs = [
            convert_real_to_embedding(all_inputs[..., i : i + 1])
            for i in self.known_regular_input_idx
            if i not in self.static_input_loc
        ]

        known_combined_layer = Lambda(tf_stack, arguments={"axis": -1})(known_regular_inputs)

        return unknown_inputs, known_combined_layer, obs_inputs, static_inputs

    def get_inputs_orig(self, all_inputs):
        """Get inputs for graph - original inputs isolation based on `encoder_steps`."""
        # Size definitions.
        encoder_steps = self.num_encoder_steps

        (unknown_inputs, known_combined_layer, obs_inputs, static_inputs,) = self.get_tft_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        if unknown_inputs is not None:
            historical_inputs = tf.keras.layers.Concatenate(axis=-1)(
                [
                    unknown_inputs[:, :encoder_steps, :],
                    known_combined_layer[:, :encoder_steps, :],
                    obs_inputs[:, :encoder_steps, :],
                ],
            )
        else:
            historical_inputs = tf.keras.layers.Concatenate(axis=-1)(
                [known_combined_layer[:, :encoder_steps, :], obs_inputs[:, :encoder_steps, :]]
            )

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        return historical_inputs, future_inputs, static_inputs

    def get_inputs_from_windows(self, all_inputs):
        """Get inputs for graph - inputs isolation based on `encoder_steps` for windowed approach with alignment.

        Since TFT uses forking sequences, it needs to mask the times of the input to use only past known data
        (:encoder_steps) and future known data (encoder_steps:)
        """
        # Size definitions.
        encoder_steps = self.num_encoder_steps

        (unknown_inputs, known_combined_layer, obs_inputs, static_inputs,) = self.get_tft_embeddings(all_inputs)

        # Isolate known and observed historical inputs.
        knowns = [unknown_inputs, known_combined_layer, obs_inputs]
        knowns_times = [inp[:, :encoder_steps, :] for inp in knowns if inp is not None]
        historical_inputs = tf.keras.layers.Concatenate(axis=-1)(knowns_times)

        # Isolate only known future inputs.
        future_inputs = known_combined_layer[:, encoder_steps:, :]

        return historical_inputs, future_inputs, static_inputs

    def build_base_tft_graph(
        self, historical_inputs, future_inputs, static_inputs, batch_dimensions=1
    ) -> Tuple[Layer, Dict]:
        """Returns graph defining layers of the TFT.

        Args:
            :batch_dimensions
            Set to 2 to try experimental feature of the model to operate on 'more batch dimensions' more naturally.
            So far the only thing that is preventing the model from being completely oblivious to the number
            of batch dimensions is the lstm layer, that seems to need a specific number of dimensions.
        """

        def static_combine_and_mask(embedding):
            """Applies variable selection network to static inputs.

            Args:
                embedding: Transformed static inputs

            Returns:
                Tensor output for variable selection network
            """

            # Add temporal features
            _, num_static, static_dim = embedding.get_shape().as_list()[-3:]

            # as flatten = tf.keras.layers.Flatten()(embedding) would work for 3D,
            # this is the way we awant to flatten for 4D:
            shape = tf.shape(embedding)
            flatten = tf.reshape(embedding, tf.concat([shape[:-2], [num_static * static_dim]], axis=-1))

            # Nonlinear transformation with gated residual network.
            mlp_outputs = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_static,
                dropout_rate=self.dropout_rate,
                use_time_distributed=False,
                additional_context=None,
            )

            sparse_weights = tf.keras.layers.Activation("softmax")(mlp_outputs)
            sparse_weights = Lambda(tf.expand_dims, arguments={"axis": -1})(sparse_weights)

            trans_emb_list = []
            for i in range(num_static):
                e = gated_residual_network(
                    embedding[:, i : i + 1, :],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=False,
                )
                trans_emb_list.append(e)

            transformed_embedding = (
                tf.keras.layers.Concatenate(axis=1)(trans_emb_list) if len(trans_emb_list) > 1 else trans_emb_list[0]
            )

            combined = tf.keras.layers.multiply([sparse_weights, transformed_embedding])

            static_vec = Lambda(K.sum, arguments={"axis": 1})(combined)

            return static_vec, sparse_weights

        if static_inputs is not None:
            static_encoder, static_weights = static_combine_and_mask(static_inputs)

            static_context_variable_selection = gated_residual_network(
                static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False,
            )
            static_context_enrichment = gated_residual_network(
                static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False,
            )
            static_context_state_h = gated_residual_network(
                static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False,
            )
            static_context_state_c = gated_residual_network(
                static_encoder, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=False,
            )

        def lstm_combine_and_mask(embedding):
            """Apply temporal variable selection networks.

            Args:
                embedding: Transformed inputs.

            Returns:
                Processed tensor outputs.
            """

            # Add temporal features
            time_steps, embedding_dim, num_inputs = embedding.get_shape().as_list()[-3:]

            batch_dimensions = tf.shape(embedding)[:-3]
            # new_shape = [-1, time_steps, embedding_dim * num_inputs]
            new_shape = tf.concat([batch_dimensions, [time_steps, embedding_dim * num_inputs]], axis=-1)
            flatten = tf.reshape(embedding, shape=new_shape)

            if static_inputs is not None:
                expanded_static_context = Lambda(tf.expand_dims, arguments={"axis": 1})(
                    static_context_variable_selection
                )
            else:
                expanded_static_context = None

            # Variable selection weights
            mlp_outputs, static_gate = gated_residual_network(
                flatten,
                self.hidden_layer_size,
                output_size=num_inputs,
                dropout_rate=self.dropout_rate,
                use_time_distributed=True,
                additional_context=expanded_static_context,
                return_gate=True,
            )

            sparse_weights = tf.keras.layers.Activation("softmax")(mlp_outputs)
            sparse_weights = Lambda(tf.expand_dims, arguments={"axis": -2})(sparse_weights)

            # Non-linear Processing & weight application
            trans_emb_list = []
            for i in range(num_inputs):
                grn_output = gated_residual_network(
                    embedding[..., i],
                    self.hidden_layer_size,
                    dropout_rate=self.dropout_rate,
                    use_time_distributed=True,
                )
                trans_emb_list.append(grn_output)

            transformed_embedding = Lambda(tf_stack, arguments={"axis": -1})(trans_emb_list)

            combined = tf.keras.layers.multiply([sparse_weights, transformed_embedding])
            temporal_ctx = Lambda(K.sum, arguments={"axis": -1})(combined)

            return temporal_ctx, sparse_weights, static_gate

        historical_features, historical_flags, _ = lstm_combine_and_mask(historical_inputs)
        future_features, future_flags, _ = lstm_combine_and_mask(future_inputs)

        # LSTM layer
        def get_lstm(return_state):
            """Returns LSTM cell initialized with default parameters."""
            lstm = tf.keras.layers.LSTM(
                self.hidden_layer_size,
                return_sequences=True,
                return_state=return_state,
                stateful=False,
                # Additional params to ensure LSTM matches CuDNN, See TF 2.0 :
                # (https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
                activation="tanh",
                recurrent_activation="sigmoid",
                recurrent_dropout=0,
                unroll=False,
                use_bias=True,
            )
            return lstm

        if static_inputs is not None:
            history_lstm, state_h, state_c = timedistributed_of_lstm_state(
                get_lstm(return_state=True),
                batch_dimensions,
                historical_features,
                initial_state=[static_context_state_h, static_context_state_c],
            )

            future_lstm = timedistributed_over_more_batch_dimensions(
                get_lstm(return_state=False), batch_dimensions, future_features, initial_state=[state_h, state_c],
            )
        else:
            history_lstm = timedistributed_over_more_batch_dimensions(
                get_lstm(return_state=False), batch_dimensions, historical_features
            )
            future_lstm = timedistributed_over_more_batch_dimensions(
                get_lstm(return_state=False), batch_dimensions, future_features
            )

        lstm_layer = tf.keras.layers.Concatenate(axis=-2)([history_lstm, future_lstm])

        # Apply gated skip connection:
        input_embeddings = tf.keras.layers.Concatenate(axis=-2)([historical_features, future_features])
        # (^ Here is the last point, where history_ and future_ properties exist (also they do merge here))

        lstm_layer, _ = apply_gating_layer(lstm_layer, self.hidden_layer_size, self.dropout_rate, activation=None)
        temporal_feature_layer = add_and_norm([lstm_layer, input_embeddings])

        # Static enrichment layers
        if static_inputs is not None:
            expanded_static_context = Lambda(tf.expand_dims, arguments={"axis": -2})(static_context_enrichment)
        else:
            expanded_static_context = None
        enriched, _ = gated_residual_network(
            temporal_feature_layer,
            self.hidden_layer_size,
            dropout_rate=self.dropout_rate,
            use_time_distributed=True,
            additional_context=expanded_static_context,
            return_gate=True,
        )

        # Decoder self attention
        self_attn_layer = InterpretableMultiHeadAttention(
            self.num_heads, self.hidden_layer_size, dropout=self.dropout_rate
        )

        def do_attention(enriched):
            mask = Lambda(get_decoder_mask)(enriched)
            x, self_att = self_attn_layer(enriched, enriched, enriched, mask=mask)
            return x, self_att

        x, self_att = timedistributed_over_more_batch_dimensions(do_attention, batch_dimensions, enriched)

        x, _ = apply_gating_layer(x, self.hidden_layer_size, dropout_rate=self.dropout_rate, activation=None)
        x = add_and_norm([x, enriched])

        # Nonlinear processing on outputs
        decoder = gated_residual_network(
            x, self.hidden_layer_size, dropout_rate=self.dropout_rate, use_time_distributed=True,
        )

        # Final skip connection
        decoder, _ = apply_gating_layer(decoder, self.hidden_layer_size, activation=None)
        transformer_layer = add_and_norm([decoder, temporal_feature_layer])

        # Attention components for explainability
        attention_components = {
            # Temporal attention weights
            "decoder_self_attn": self_att,
            # Static variable selection weights
            "static_flags": static_weights[..., 0] if static_inputs is not None else [],
            # Variable selection weights of past inputs
            "historical_flags": historical_flags[..., 0, :],
            # Variable selection weights of future inputs
            "future_flags": future_flags[..., 0, :],
        }

        return transformer_layer, attention_components

    def get_model(self) -> tf.keras.Model:
        all_inputs = tf.keras.Input(shape=self.input_shape, name="input")

        # 1. Use for orig inputs format
        # historical_inputs, future_inputs, static_inputs = self.get_inputs_orig(all_inputs)
        # 2. Use for windowed inputs approach with data alignment
        historical_inputs, future_inputs, static_inputs = self.get_inputs_from_windows(all_inputs)

        transformer_layer, attention_components = self.build_base_tft_graph(
            historical_inputs, future_inputs, static_inputs
        )

        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_dim, activation=self.last_activation), name="output",
        )(transformer_layer[..., self.num_encoder_steps :, :])

        self._attention_components = attention_components

        return tf.keras.models.Model(inputs=all_inputs, outputs=outputs)

    def create_named_inputs(self, past_size, future_size, single_sequence=False, known_size=None):
        static_emb = None
        obs_emb = None
        unknown_emb = None
        known_emb = None
        static = None
        observed = None
        unknown = None
        known = None
        future = None

        n_static = self.static_lookup_size
        if len(self.static_input_loc) > 0:
            logger.warning(
                "Static inputs present, but we are using named (packed) representation and static inputs"
                " are not time-dependent."
            )
        n_observed = len(self.input_obs_loc)
        n_unknown = len(self.unknowns)
        n_known = len(self.knowns)

        if n_static > 0:
            static = tf.keras.Input(shape=[n_static], name="static")
            static_inputs = [
                tf.keras.layers.Dense(self.hidden_layer_size)(static[..., i : i + 1]) for i in range(n_static)
            ]
            static_emb = Lambda(tf_stack, arguments={"axis": 1})(static_inputs)

        def convert_real_to_embedding(x):
            """Applies linear transformation for time-varying inputs."""
            return tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.hidden_layer_size))(x)

        if n_observed > 0:
            observed = tf.keras.Input(shape=[past_size, n_observed], name="observed")
            obs_inputs = [convert_real_to_embedding(observed[..., i : i + 1]) for i in range(n_observed)]
            obs_emb = Lambda(tf_stack, arguments={"axis": -1})(obs_inputs)

        if n_unknown > 0:
            unknown = tf.keras.Input(shape=[past_size, n_unknown], name="unknown")
            unknown_inputs = [convert_real_to_embedding(unknown[..., i : i + 1]) for i in range(n_unknown)]
            unknown_emb = Lambda(tf_stack, arguments={"axis": -1})(unknown_inputs)

        if n_known > 0:
            if single_sequence:
                known_size = known_size
            else:
                known_size = past_size
            known = tf.keras.Input(shape=[known_size, n_known], name="known")
            known_inputs = [convert_real_to_embedding(known[..., i : i + 1]) for i in range(n_known)]
            known_emb = Lambda(tf_stack, arguments={"axis": -1})(known_inputs)
        else:
            if single_sequence:
                raise ValueError("Known inputs must be specified since they are used also as future inputs.")

        if single_sequence:
            if unknown_emb is None and obs_emb is None and past_size is None:
                raise ValueError("Not enough information to infer sequence sizes for TFT")
            if self.get_future_fork_size() is None:
                raise ValueError("For single_sequence, get_future_fork_size() needs to be specified.")

            seq_non_future = [item for item in [unknown_emb, obs_emb] if item is not None]
            if seq_non_future:
                # if we have any non-known sequence, then we cut the knowns at the same size, making it into past-knowns
                inferred_future_cut = get_shape_fixed_if_possible(seq_non_future[0])[-3]
            else:
                inferred_future_cut = (
                    get_shape_fixed_if_possible(known_emb)[-3] - past_size - self.get_future_fork_size()
                )
            inferred_past_cut = (
                self.num_encoder_steps
            )  # the future data need to be offset by the future prediction window
            known_emb_past = known_emb[:, 0:inferred_future_cut, ...]
            known_emb_future = known_emb[:, inferred_past_cut:, ...]

            # assert len in timesteps of known_emb = known_emb_past (or any other past sequence) + future_size
        else:
            # Isolate only known future inputs.
            future = tf.keras.Input(shape=[future_size, n_known], name="future-known")
            future_inputs = [convert_real_to_embedding(future[..., i : i + 1]) for i in range(n_known)]
            known_emb_future = Lambda(tf_stack, arguments={"axis": -1})(future_inputs)
            known_emb_past = known_emb

        # Isolate known and observed historical inputs.
        knowns_times = [inp for inp in [unknown_emb, known_emb_past, obs_emb] if inp is not None]
        historical_inputs = tf.keras.layers.Concatenate(axis=-1)(knowns_times)

        all_inputs = [inp for inp in [unknown, known, observed, static, future] if inp is not None]

        return historical_inputs, known_emb_future, static_emb, all_inputs

    def get_model_named_inputs(self):
        """Creates TFT model with named inputs.

        The goal is to not force TFT to unpack parameters from (windowed) blobs, but send everything already separated.
        """
        (historical_inputs, future_emb, static_emb, all_inputs,) = self.create_named_inputs(
            self.num_encoder_steps, self.get_future_fork_size()
        )

        transformer_layer, attention_components = self.build_base_tft_graph(historical_inputs, future_emb, static_emb)

        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_dim, activation=self.last_activation), name="output",
        )(transformer_layer[..., self.num_encoder_steps :, :])

        self._attention_components = attention_components

        return tf.keras.models.Model(inputs=all_inputs, outputs=outputs)

    def get_model_vectorized(self, model_capable_vectorize=True, single_sequence=True):
        """Creates a vectorized version of the model, that should behave like normal sequential (CNN/LSTM) model.
        (And not require windows for training.)

        Args:
            model_capable_vectorize: We trust the mechanism inside the model to be able to vectorize inside
            single_sequence: If set to True, known and future-known inputs will become one and the model will
                select past and future inputs cleverly by itself.
        Both those features are parametrized to allow for experimentation and stepback in case of failures.

        Explanation:
            2 timeseries: H 'historical', F 'future' (some features from historical can be in the future )

            H ############################## (F is longer, since future known inputs need to be .. known)
            F -----############################## (- means, that this input will never be used and so is omitted)

            windowed approach takes history len or embedding len from H and then the needed future inputs from F:
            H XXXXXXXXXX####################
            F -----#####XXXXX####################
                       ^ at each timepoint like this

            ... and so if we would be able to transform from this:
            H 123456789012345678901234567890
            F -----FGHIJKLMNOPQRSTUVWXYZABCDEFGHI

            ... into this (or transposed):

            1234...901 ............ then we have all the windows in the vertical direction
            2345...012 ............ and we can somehow make the computation batched
            3456...123 ............ note, that it is just a windowing technique applied on two streams:
            4567...234
            5678...345
            6789...456
            7890...567
            8901...678
            9012...789
            0123...890
            KLMN...CDE
            LMNO...DEF
            MNOP...EFG
            NOPQ...FGH
            OPQR...GHI


            technical possibilities:
            vectorize slice, roll (not supported atm.), matrix multiplication (too heavy?), gather (used)
            or ideas from https://www.tensorflow.org/probability/api_docs/python/tfp/math/fill_triangular
        """
        (historical_inputs, future_emb, static_emb, all_inputs,) = self.create_named_inputs(
            None,
            None,
            single_sequence,
            # None instead of self.num_encoder_steps, self.input_shape[0] - self.num_encoder_steps to accept sequences
        )

        if model_capable_vectorize:
            # windowing described in the docstring ^
            historical_windowed = windowing_mechanism(
                historical_inputs, batch_dims=1, window_len=self.num_encoder_steps
            )
            # since the insides of the model are not easily used on 4 dim sequence with first two dimensions being batch
            # we need to squash the dimensions into the batch dimension:

            # now we must repeat the batch dimension for static inputs to match the new batch dimension of "squashed"
            # static_emb_repeated = repeat_multiply_batch_dimension(static_emb, tf.shape(historical_windowed)[1])
            # static_emb = tf.repeat(tf.expand_dims(static_emb, 1), tf.shape(historical_windowed)[1], axis=1)

            future_windowed = windowing_mechanism(future_emb, batch_dims=1, window_len=self.get_future_fork_size())

            transformer_layer, attention_components = self.build_base_tft_graph(
                historical_windowed, future_windowed, static_emb, batch_dimensions=2
            )
        else:
            # windowing described in the docstring ^
            historical_windowed = windowing_mechanism(
                historical_inputs, batch_dims=1, window_len=self.num_encoder_steps
            )
            # since the insides of the model are not easily used on 4 dim sequence with first two dimensions being batch
            # we need to squash the dimensions into the batch dimension:
            (historical_windowed_squashed, historical_windowed_orig_size,) = squash_batch_dimensions(
                historical_windowed, batch_dims=2
            )

            # now we must repeat the batch dimension for static inputs to match the new batch dimension of "squashed"
            static_emb_repeated = repeat_multiply_batch_dimension(static_emb, tf.shape(historical_windowed)[1])

            future_windowed = windowing_mechanism(future_emb, batch_dims=1, window_len=self.get_future_fork_size())
            (future_windowed_squashed, future_windowed_orig_size,) = squash_batch_dimensions(
                future_windowed, batch_dims=2
            )

            (transformer_layer_squashed, attention_components,) = self.build_base_tft_graph(
                historical_windowed_squashed, future_windowed_squashed, static_emb_repeated,
            )
            # back to 4D: (from 3D squashed to batch dimension)
            transformer_layer = unsquash_batch_dimensions(transformer_layer_squashed, historical_windowed_orig_size)

        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(self.output_dim, activation=self.last_activation), name="output",
        )(transformer_layer[..., self.num_encoder_steps :, :])

        self._attention_components = attention_components

        return tf.keras.models.Model(inputs=all_inputs, outputs=outputs)
