import argparse
import json
import joblib
import numpy as np
import os
import pandas as pd
import re
import tensorflow as tf
import time

import dataset
import trees
import utils
from sklearn.tree import DecisionTreeClassifier




# TensorFlow 2.x uses eager execution by default — no need to enable it manually

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--sigma', type=float, default=1.0)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--distance_weight', type=float, default=0.01)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--opt', type=str, default='adam', help='Options: adam or gd')
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--data_name', type=str, required=True)
parser.add_argument('--model_type', type=str, required=True)
parser.add_argument('--distance_function', type=str, required=True)
args = parser.parse_args()

# Validate arguments
if args.sigma == 0 or args.temperature < 0 or args.distance_weight < 0 or args.lr < 0:
    parser.error("Invalid hyperparameter values.")
for arg_name in ['opt', 'model_name', 'data_name', 'model_type']:
    if getattr(args, arg_name).strip() == '':
        parser.error(f"--{arg_name} cannot be empty.")

# Load model
model = joblib.load(f'models/{args.model_name}')

# Load data
feat_columns, feat_matrix, feat_missing_mask = dataset.read_tsv_file(f'data/{args.data_name}.tsv')
n_examples = feat_matrix.shape[0]
n_class = len(model.classes_)

# Prepare input features
feat_input = feat_matrix[:, :-1]
median_values = np.median(feat_input, axis=0)
mad = np.mean(np.abs(feat_input - median_values[None, :]), axis=0)

# Predict ground truth
ground_truth = model.predict(feat_input)
class_index = np.array([np.where(model.classes_ == label)[0][0] for label in ground_truth], dtype=np.int64)
class_index = tf.constant(class_index, dtype=tf.int64)
example_range = tf.constant(np.arange(n_examples, dtype=np.int64))
example_class_index = tf.stack((example_range, class_index), axis=1)

# Load training data for Mahalanobis distance
train_name = re.sub('test', 'train', args.data_name)
train_data = pd.read_csv(f'data/{train_name}.tsv', sep='\t', index_col=0)
x_train = np.array(train_data.iloc[:, :-1])
covar = utils.tf_cov(x_train)
inv_covar = tf.linalg.inv(covar)

# Initialize perturbed features
perturbed = tf.Variable(initial_value=feat_input, name='perturbed_features', trainable=True)

# Model conversion functions
def convert_model(sigma, temperature):
    return trees.get_prob_classification_forest(model, feat_columns, perturbed, sigma=sigma, temperature=temperature)

def prob_from_input(perturbed, sigma, temperature):
    if isinstance(model, DecisionTreeClassifier):
        return trees.get_prob_classification_tree(model, feat_columns, perturbed, sigma=sigma)
    else:
        return trees.get_prob_classification_forest(model, feat_columns, perturbed, sigma=sigma, temperature=temperature)

# Optimizer setup
if args.opt == 'adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
elif args.opt == 'gd':
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.lr)

# Output path
output_root = f'hyperparameter_tuning/{args.distance_function}/{args.data_name}/{args.model_type}/perturbs_{args.opt}_sigma{args.sigma}_temp{args.temperature}_dweight{args.distance_weight}_lr{args.lr}'

# Optimization loop
num_iter = 100
sigma = np.full(n_examples, args.sigma)
temperature = np.full(n_examples, args.temperature)
distance_weight = np.full(n_examples, args.distance_weight)
to_optimize = [perturbed]
indicator = np.ones(n_examples)
best_perturb = np.zeros(perturbed.shape)
best_distance = np.full(n_examples, 1000.)
best_relative_distance = np.full(n_examples, 1000.)
perturb_iteration_found = np.full(n_examples, 1000 * num_iter, dtype=np.int64)
average_distance = np.zeros(num_iter)

start_time = time.time()
with utils.safe_open(output_root + '.txt', 'w') as fout:
    fout.write(f'{args.model_name} {args.opt} {args.distance_function} --sigma={args.sigma} --temp={args.temperature} --distance_weight={args.distance_weight} --lr={args.lr}\n')

    for i in range(num_iter):
        with tf.GradientTape() as tape:
            p_model = utils.filter_hinge_loss(n_class, indicator, perturbed, sigma, temperature, prob_from_input)
            approx_prob = tf.gather_nd(p_model, example_class_index)

            if args.distance_function == 'euclidean':
                distance = utils.safe_euclidean(perturbed - feat_input, axis=1)
            elif args.distance_function == 'cosine':
                distance = utils.safe_cosine(perturbed, feat_input)
            elif args.distance_function == 'l1':
                distance = utils.safe_l1(perturbed - feat_input)
            elif args.distance_function == 'mahal':
                distance = utils.safe_mahal(perturbed - feat_input, inv_covar)

            hinge_approx_prob = indicator * approx_prob
            loss = tf.reduce_mean(hinge_approx_prob + distance_weight * distance)

        grad = tape.gradient(loss, to_optimize)
        optimizer.apply_gradients(zip(grad, to_optimize))
        perturbed.assign(tf.clip_by_value(perturbed, 0.0, 1.0))

        if args.distance_function == 'euclidean':
            true_distance = utils.true_euclidean(perturbed - feat_input, axis=1)
        elif args.distance_function == 'cosine':
            true_distance = utils.true_cosine(perturbed, feat_input)
        elif args.distance_function == 'l1':
            true_distance = utils.true_l1(perturbed - feat_input)
        elif args.distance_function == 'mahal':
            true_distance = utils.true_mahal(perturbed - feat_input, inv_covar)




        cur_predict = model.predict(perturbed.numpy())
        indicator = np.equal(ground_truth, cur_predict).astype(np.float64)
        idx_flipped = np.argwhere(indicator == 0).flatten()

        mask_flipped = np.not_equal(ground_truth, cur_predict)
        perturb_iteration_found[idx_flipped] = np.minimum(i + 1, perturb_iteration_found[idx_flipped])

        distance_numpy = true_distance.numpy()
        mask_smaller_dist = np.less(distance_numpy, best_distance)

        relative_distance = distance_numpy / (np.linalg.norm(feat_input, axis=1) + 1e-10)
        

        temp_dist = best_distance.copy()
        temp_dist[mask_flipped] = distance_numpy[mask_flipped]
        best_distance[mask_smaller_dist] = temp_dist[mask_smaller_dist]

        best_relative_distance[mask_flipped] = relative_distance[mask_flipped]

        temp_perturb = best_perturb.copy()
        temp_perturb[mask_flipped] = perturbed.numpy()[mask_flipped]
        best_perturb[mask_smaller_dist] = temp_perturb[mask_smaller_dist]

        fout.write(f'iteration: {i}\n')
        fout.write(f'loss: {loss.numpy()} ')
        fout.write(f'unchanged: {np.sum(indicator)} ')
        fout.write(f'prob: {tf.reduce_mean(approx_prob).numpy()} ')
        fout.write(f'mean dist: {tf.reduce_mean(distance).numpy()} ')
        fout.write(f'mean relative dist: {np.mean(relative_distance)} ')
        fout.write(f'sigma: {np.amax(sigma)} ')
        fout.write(f'temp: {np.amax(temperature)}\n')

        end_time = time.time()
        unchanged_ever = best_distance[best_distance == 1000.]
        counterfactual_examples = best_distance[best_distance != 1000.]
        average_distance[i] = np.mean(counterfactual_examples)

        counterfactual_relative = best_relative_distance[best_relative_distance != 1000.]
        mean_relative_distance = np.mean(counterfactual_relative)

        num_relative_lt_1 = np.sum(best_relative_distance < 1.0)
        percent_relative_lt_1 = 100.0 * num_relative_lt_1 / n_examples

        fout.write(f'Unchanged ever: {len(unchanged_ever)}\n')
        if len(unchanged_ever) == 0:
            fout.write(f'Mean {args.distance_function} dist for cf example v1: {tf.reduce_mean(best_distance)}\n')
            fout.write(f'Mean {args.distance_function} dist for cf example v2: {np.mean(counterfactual_examples)}\n')
            fout.write(f'Mean relative dist for cf example: {mean_relative_distance}\n')
        else:
            fout.write('Not all instances have counterfactual examples!!\n')
        fout.write('-------------------------- \n')

    fout.write(f'Finished in: {np.round(end_time - start_time, 2)}sec \n')

perturb_iteration_found[perturb_iteration_found == 1000 * num_iter] = 0

cf_stats = {
    'dataset': args.data_name,
    'model_type': args.model_type,
    'opt': args.opt,
    'distance_function': args.distance_function,
    'sigma': args.sigma,
    'temp': args.temperature,
    'dweight': args.distance_weight,
    'lr': args.lr,
    'unchanged_ever': len(unchanged_ever),
    'mean_dist': np.mean(counterfactual_examples),
    'mean_relative_dist': mean_relative_distance,
    'percent_relative_dist_lt_1': percent_relative_lt_1
}


with utils.safe_open(output_root + '_cf_stats.txt', 'w') as gsout:
	json.dump(cf_stats, gsout)

# Output results

df_dist = pd.DataFrame({
    "id": range(n_examples),
    "best_distance": best_distance,
    "best_relative_distance": best_relative_distance
})

df_perturb = pd.DataFrame(best_perturb, columns=feat_columns[:-1])
df = pd.concat([df_dist, df_perturb], axis=1)

df.to_csv(output_root + '.tsv', sep='\t')
print("Finished!! ~{} sec".format(np.round(end_time - start_time), 2))
