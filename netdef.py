import tensorflow as tf

WEIGHTS_INIT_STDEV = .1

def shortcut_interpolation(image, sc, factor):
    # Valid factor range [0.0, 2.0], detailed interpolation are showing as follow:
    #   (1) When factor is in [0.0, 1.0], the stroke are combined with (1 - factor) * 256 + factor * 512
    #   (2) When factor is in [1.0, 2.0], the stroke are combined with (2 - factor) * 512 + (factor - 1) * 768
    # As a consequence of this design, the stroke will grow from 256 to 768 when factor grows from 0 to 2
    alpha = tf.cond(sc[0],
        lambda: tf.cond(sc[1],
            lambda: tf.maximum(0.0, 1.0 - factor),
            lambda: tf.constant(1.0)
        ),
        lambda: tf.constant(0.0)
    )
    beta = tf.cond(sc[0],
        lambda: tf.cond(sc[1],
            lambda: 1.0 - tf.sign(factor - 1.0) * (factor - 1.0),
            lambda: tf.constant(0.0)
        ),
        lambda: tf.cond(sc[1],
            lambda: tf.constant(1.0),
            lambda: tf.constant(0.0)
        )
    )
    gamma = tf.cond(sc[0],
        lambda: tf.cond(sc[1],
            lambda: tf.maximum(factor - 1.0, 0.0),
            lambda: tf.constant(0.0)
        ),
        lambda: tf.cond(sc[1],
            lambda: tf.constant(0.0),
            lambda: tf.constant(1.0)
        )
    )
    conv1 = _conv_layer(image, 16, 3, 1)
    conv2 = _conv_layer(conv1, 32, 3, 2)
    conv3 = _conv_layer(conv2, 48, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid4_1 = _residual_block(resid4, 3)
    resid5 = alpha * resid3 + beta * resid4 + gamma * resid4_1
    conv_t1 = _conv_tranpose_layer(resid5, 32, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 16, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255./2
    return preds

def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net

def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 48, filter_size, 1)
    return net + _conv_layer(tmp, 48, filter_size, 1, relu=False)

def _instance_norm(net, train=True):
    in_channels = net.get_shape().as_list()[3]
    var_shape = [in_channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net-mu) / (sigma_sq + epsilon)**(.5)
    return scale * normalized + shift

def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)
    batch_size = net.get_shape().as_list()[0]
    channels = net.get_shape().as_list()[3]

    net_shape = tf.shape(net)
    rows, cols = net_shape[1], net_shape[2]

    new_rows, new_cols = rows * strides, cols * strides
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, new_shape, strides_shape, padding='SAME')
    net = tf.reshape(net, [batch_size, new_rows, new_cols, num_filters])
    net = _instance_norm(net)
    return tf.nn.relu(net)

def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    in_channels = net.get_shape().as_list()[3]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init

def gram_matrix(activations):
    height = tf.shape(activations)[1]
    width = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix = tf.transpose(activations, [0, 3, 1, 2])
    gram_matrix = tf.reshape(gram_matrix, [num_channels, width * height])
    gram_matrix = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix
