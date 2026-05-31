import tensorflow as tf


def channel_wise_attention(feature_map, H, W, C, weight_decay=0.00004, scope='', reuse=None):
    """Compatibility-equivalent channel-wise attention.

    The original implementation expands channel weights with a very long
    tf.concat list. Under TensorFlow 1.15 on this machine, Grappler reports a
    concat self-cycle. tf.tile expresses the same broadcast without that graph
    pattern.
    """
    with tf.variable_scope(scope, 'ChannelWiseAttention', reuse=reuse):
        weight = tf.get_variable(
            "weight",
            [C, C],
            dtype=tf.float32,
            initializer=tf.initializers.orthogonal,
            regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
        )
        bias = tf.get_variable(
            "bias",
            [C],
            dtype=tf.float32,
            initializer=tf.initializers.zeros,
        )

        transpose_feature_map = tf.transpose(
            tf.reduce_mean(feature_map, [1, 2], keep_dims=True),
            perm=[0, 3, 1, 2],
        )
        channel_wise_attention_fm = tf.matmul(
            tf.reshape(transpose_feature_map, [-1, C]),
            weight,
        ) + bias
        channel_wise_attention_fm = tf.nn.sigmoid(channel_wise_attention_fm)
        attention = tf.reshape(channel_wise_attention_fm, [-1, 1, 1, C])
        attention = tf.tile(attention, [1, H, W, 1])
        return attention * feature_map
