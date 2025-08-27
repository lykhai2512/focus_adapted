import tensorflow as tf
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

def _exact_activation_by_index(feat_input, feat_index, threshold):
    boolean_act = tf.math.greater(feat_input[:, feat_index], threshold)
    return tf.logical_not(boolean_act), boolean_act

def _approx_activation_by_index(feat_input, feat_index, threshold, sigma):
    activation = tf.math.sigmoid((feat_input[:, feat_index] - threshold) * sigma)
    return 1.0 - activation, activation

def _double_activation_by_index(feat_input, feat_index, threshold, sigma):
    e_l, e_r = _exact_activation_by_index(feat_input, feat_index, threshold)
    a_l, a_r = _approx_activation_by_index(feat_input, feat_index, threshold, sigma)
    return (e_l, a_l), (e_r, a_r)

def _split_node_by_index(node, feat_input, feat_index, threshold, sigma):
    e_o, a_o = node
    (e_l, a_l), (e_r, a_r) = _double_activation_by_index(feat_input, feat_index, threshold, sigma)
    return (tf.logical_and(e_l, e_o), a_l * a_o), (tf.logical_and(e_r, e_o), a_r * a_o)

def _split_approx(node, feat_input, feat_index, threshold, sigma):
    if node is None:
        node = tf.constant(1.0, dtype=tf.float64)
    l_n, r_n = _approx_activation_by_index(feat_input, feat_index, threshold, sigma)
    return node * l_n, node * r_n

def _split_exact(node, feat_input, feat_index, threshold, sigma):
    if node is None:
        node = tf.constant(True)
    l_n, r_n = _exact_activation_by_index(feat_input, feat_index, threshold)
    return tf.logical_and(node, l_n), tf.logical_and(node, r_n)

def _parse_class_tree(tree, feat_columns, feat_input, split_function):
    n_classes = len(tree.classes_)
    n_nodes = tree.tree_.node_count
    children_left = tree.tree_.children_left
    children_right = tree.tree_.children_right
    feature = tree.tree_.feature
    threshold = tree.tree_.threshold
    values = tree.tree_.value

    nodes = [None] * n_nodes
    leaf_nodes = [[] for _ in range(n_classes)]

    for i in range(n_nodes):
        cur_node = nodes[i]
        if children_left[i] != children_right[i]:
            l_n, r_n = split_function(cur_node, feat_input, feature[i], threshold[i])
            nodes[children_left[i]] = l_n
            nodes[children_right[i]] = r_n
        else:
            max_class = np.argmax(values[i])
            leaf_nodes[max_class].append(cur_node)

    return leaf_nodes

def get_prob_classification_tree(tree, feat_columns, feat_input, sigma):
    def split_function(node, feat_input, feat_index, threshold):
        return _split_approx(node, feat_input, feat_index, threshold, sigma)

    leaf_nodes = _parse_class_tree(tree, feat_columns, feat_input, split_function)

    if tree.tree_.node_count > 1:
        n_classes = len(tree.classes_)
        out_l = [sum(leaf_nodes[c_i]) for c_i in range(n_classes)]
        stacked = tf.stack(out_l, axis=-1)
    else:
        only_class = tree.predict(tf.reshape(feat_input[0, :], shape=(1, -1)))[0]
        n_input = tf.shape(feat_input)[0]
        correct_class = tf.ones(n_input, dtype=tf.float64)
        incorrect_class = tf.zeros(n_input, dtype=tf.float64)

        if only_class == 1.0:
            class_labels = [incorrect_class, correct_class]
        elif only_class == 0.0:
            class_labels = [correct_class, incorrect_class]
        else:
            raise ValueError("Unexpected class label in single-node tree.")

        stacked = tf.stack(class_labels, axis=1)

    return stacked

def get_exact_classification_tree(tree, feat_columns, feat_input):
    leaf_nodes = _parse_class_tree(tree, feat_columns, feat_input, _split_exact)
    n_classes = len(tree.classes_)
    out_l = [tf.reduce_any(tf.stack(leaf_nodes[c_i]), axis=0) for c_i in range(n_classes)]
    return tf.cast(tf.stack(out_l, axis=-1), dtype=tf.float64)

def get_prob_classification_forest(model, feat_columns, feat_input, sigma=10.0, temperature=1.0):
    tree_outputs = [get_prob_classification_tree(est, feat_columns, feat_input, sigma)
                    for est in model.estimators_[:100]]

    if isinstance(model, AdaBoostClassifier):
        weights = model.estimator_weights_
    elif isinstance(model, RandomForestClassifier):
        weights = np.full(len(tree_outputs), 1.0 / len(tree_outputs))
    else:
        raise TypeError("Unsupported model type.")

    logits = sum(w * tree for w, tree in zip(weights, tree_outputs))

    if isinstance(temperature, (float, int)):
        expits = tf.exp(temperature * logits)
    else:
        expits = tf.exp(temperature[:, None] * logits)

    softmax = expits / tf.reduce_sum(expits, axis=1, keepdims=True)
    return softmax
