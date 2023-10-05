import numpy as np
from tqdm import tqdm, trange
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, Model

import sklearn.metrics
from sklearn.model_selection import train_test_split

import argparse
import datetime
import os
import sys; sys.path.append('../'); sys.path.append('../..')
import utils
import models

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
    default='object_sorting_datasets/product_structure_object_sort_dataset.npy')
parser.add_argument('--n_epochs', type=int, default=400)
parser.add_argument('--early_stopping', type=bool, default=False)
parser.add_argument('--train_size', type=int, default=3000)
parser.add_argument('--out_dir', type=str, default='robustness_results')
parser.add_argument('--out_dir_addtime', type=bool, default=True)
parser.add_argument('--seed', type=int, default=314159)
parser.add_argument('--wandb_project_name', type=str, default=None)

args = parser.parse_args()

out_dir = args.out_dir
if args.out_dir_addtime:
    datetimestr = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = f'{out_dir}_{datetimestr}'
out_dir = f'results/{out_dir}'

os.mkdir(out_dir)
os.mkdir(f'{out_dir}/models')

# set up W&B logging
wandb_project_name = args.wandb_project_name

if wandb_project_name:
    import wandb
    wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)


seed = args.seed

def create_callbacks(monitor='loss'):
    callbacks = []
    if wandb_project_name:
        callbacks.append(wandb.keras.WandbMetricsLogger(log_freq='epoch'))

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto', restore_best_weights=True))

    return callbacks

from transformer_modules import TeacherForcingAccuracy
teacher_forcing_acc_metric = TeacherForcingAccuracy(ignore_class=None)
metrics = [teacher_forcing_acc_metric]

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': args.n_epochs, 'batch_size': 512}

# region evaluation code
def evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False):
    
    n = len(source_test)
    output = np.zeros(shape=(n, (seqs_length+1)), dtype=int)
    output[:,0] = start_token
    for i in range(seqs_length):
        predictions = model((source_test, output[:, :-1]), training=False)
        predictions = predictions[:, i, :]
        predicted_id = tf.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    elementwise_acc = (np.mean(output[:,1:] == labels_test))
    acc_per_position = [np.mean(output[:, i+1] == labels_test[:, i]) for i in range(seqs_length)]
    seq_acc = np.mean(np.all(output[:,1:]==labels_test, axis=1))


    teacher_forcing_acc = teacher_forcing_acc_metric(labels_test, model([source_test, target_test])).numpy()
    teacher_forcing_acc_metric.reset_state()

    if print_:
        print('element-wise accuracy: %.2f%%' % (100*elementwise_acc))
        print('full sequence accuracy: %.2f%%' % (100*seq_acc))
        print('teacher-forcing accuracy:  %.2f%%' % (100*teacher_forcing_acc))


    return_dict = {
        'elementwise_accuracy': elementwise_acc, 'full_sequence_accuracy': seq_acc,
        'teacher_forcing_accuracy': teacher_forcing_acc, 'acc_by_position': acc_per_position
        }

    return return_dict

def log_to_wandb(model, evaluation_dict):
    acc_by_position_table = wandb.Table(
        data=[(i, acc) for i, acc in enumerate(evaluation_dict['acc_by_position'])], 
        columns=["position", "element-wise accuracy at position"])

    evaluation_dict['acc_by_position'] = wandb.plot.line(
        acc_by_position_table, "position", "element-wise accuracy at position",
        title="Element-wise Accuracy By Position")

    wandb.log(evaluation_dict)

#endregion

#region load data
task1_data = np.load(args.data_path, allow_pickle=True).item()

objects, seqs, sorted_seqs, object_seqs, target, labels, start_token = (task1_data['objects'], task1_data['seqs'], task1_data['sorted_seqs'], task1_data['object_seqs'], \
    task1_data['target'], task1_data['labels'], task1_data['start_token'])

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, target_train, target_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, target, labels, test_size=0.2, random_state=seed)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, target_train, target_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, target_train, labels_train, test_size=val_size/(1-test_size), random_state=seed)


X_train = object_seqs_train[:args.train_size], target_train[:args.train_size]
y_train = labels_train[:args.train_size]

X_val = object_seqs_val, target_val
y_val = labels_val

seqs_length = seqs.shape[1]

models.update_model_kwargs(seqs_length)

test_data = {'objects': objects, 
    'seqs': seqs_test, 'sorted_seqs': sorted_seqs_test,
    'target': target_test, 'labels': labels_test}
np.save(f'{out_dir}/test_data.npy', test_data)

#endregion

#region train models
utils.print_section('TRAINING MODELS')

model_names = ['transformer', 'relational abstractor', 
    'symbolic abstractor', 'simple abstractor', 'ablation model']

for model_name in model_names:
    utils.print_section(f'TRAINING {model_name.upper()}')
    if wandb_project_name:
        run = wandb.init(project=wandb_project_name, name=model_name,
            config={'train size': args.train_size})

    argsort_model = models.create_model(model_name)

    argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
    argsort_model((object_seqs_train[:32], target_train[:32]));

    argsort_model.summary()

    history = argsort_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=1, callbacks=create_callbacks(), **fit_kwargs)

    eval_dict = evaluate_seq2seq_model(argsort_model, object_seqs_test, target_test, labels_test, 
        start_token=start_token, print_=True)

    if wandb_project_name:
        log_to_wandb(argsort_model, eval_dict)

    model_path = f'{out_dir}/models/{model_name}.h5'
    argsort_model.save_weights(model_path)
    if wandb_project_name:
        trained_model_artifact = wandb.Artifact(
            model_name, metadata=models.get_model_kwargs(model_name),
            description='model weights', type="model")

        trained_model_artifact.add_file(model_path)
        run.log_artifact(trained_model_artifact)



# Output Robustness Data

robustness_data_agg_df = pd.concat(robustness_data_dataframes)
robustness_data_agg_df.to_csv(f'{out_dir}/robustness_data.csv')
