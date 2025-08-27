import tensorflow as tf
import numpy as np
import os
import errno

def filter_hinge_loss(n_class, mask_vector, feat_input, sigma, temperature, model_fn):
    n_input = tf.shape(feat_input)[0]

    if not np.any(mask_vector):
        return tf.zeros((n_input, n_class), dtype=tf.float64)

    filtered_input = tf.boolean_mask(feat_input, mask_vector)

    if not isinstance(sigma, (float, int)):
        sigma = tf.boolean_mask(sigma, mask_vector)
    if not isinstance(temperature, (float, int)):
        temperature = tf.boolean_mask(temperature, mask_vector)

    filtered_loss = model_fn(filtered_input, sigma, temperature)

    indices = tf.where(mask_vector)
    zero_loss = tf.zeros((n_input, n_class), dtype=tf.float64)

    hinge_loss = tf.tensor_scatter_nd_add(zero_loss, indices, filtered_loss)
    return hinge_loss

def safe_euclidean(x, epsilon=1e-10, axis=-1):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis) + epsilon)

def true_euclidean(x, axis=-1):
    return tf.sqrt(tf.reduce_sum(tf.square(x), axis=axis))

def safe_cosine(x1, x2, epsilon=1e-10):
    normalize_x1 = tf.nn.l2_normalize(x1, axis=1)
    normalize_x2 = tf.nn.l2_normalize(x2, axis=1)
    cosine_similarity = tf.reduce_sum(normalize_x1 * normalize_x2, axis=1)
    cosine_distance = 1.0 - cosine_similarity + epsilon
    return tf.cast(cosine_distance, tf.float64)

def true_cosine(x1, x2, axis=-1):
    normalize_x1 = tf.nn.l2_normalize(x1, axis=1)
    normalize_x2 = tf.nn.l2_normalize(x2, axis=1)
    cosine_similarity = tf.reduce_sum(normalize_x1 * normalize_x2, axis=axis)
    cosine_distance = 1.0 - cosine_similarity
    return tf.cast(cosine_distance, tf.float64)

def safe_l1(x, epsilon=1e-10, axis=1):
    return tf.reduce_sum(tf.abs(x), axis=axis) + epsilon

def true_l1(x, axis=1):
    return tf.reduce_sum(tf.abs(x), axis=axis)

def tf_cov(x):
    x = tf.cast(x, tf.float64)
    mean_x = tf.reduce_mean(x, axis=0, keepdims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_x)
    vx = tf.matmul(tf.transpose(x), x) / tf.cast(tf.shape(x)[0], tf.float64)
    return vx - mx

def safe_mahal(x, inv_covar, epsilon=1e-10):
    x = tf.cast(x, tf.float64)
    return tf.reduce_sum(tf.multiply(tf.matmul(x + epsilon, inv_covar), x + epsilon), axis=1)

def true_mahal(x, inv_covar):
    x = tf.cast(x, tf.float64)
    return tf.reduce_sum(tf.multiply(tf.matmul(x, inv_covar), x), axis=1)

def mkdir_p(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise

def safe_open(path, mode):
    mkdir_p(os.path.dirname(path))
    return open(path, mode)