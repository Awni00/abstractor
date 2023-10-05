# LEARNING CURVES AND ABSTRACTER GENERALIZATION: RANDOM OBJECT SORTING WITH SEQUENCE-TO-SEQUENCE ABSTRACTERS
# We generate random objects (as gaussian vectors) and associate an ordering to them.
# We train abstracter models to learn how to sort these objects
# To test the generalization of abstracters, we first train one on another object-sorting task, 
# then fix the abstracter module's weights and train the encoder/decoder
# The models do 'argsorting', meaning they predict the argsort of the sequnce.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
from models import (AbstractorOrderRelation, AutoregressiveSimpleAbstractor, 
    reload_rel_model, initialize_with_rel_model, reload_argsort_model,
    rel_model_kwargs, decoder_kwargs)
from evaluation_utils import evaluate_seq2seq_model, log_to_wandb
import utils

# region SETUP

seed = None

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--training_mode', default='none', type=str,
    choices=('end-to-end', 'use-rel-model', 'use-rel-model-decoder'),
    help='how to train and use pre-training task')
parser.add_argument('--eval_task_data_path', default='object_sorting_datasets/task1_object_sort_dataset.npy', 
    type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--n_epochs', default=200, type=int, help='number of epochs to train each model for')
parser.add_argument('--early_stopping', default=True, type=bool, help='whether to use early stopping')
parser.add_argument('--min_train_size', default=500, type=int, help='minimum training set size')
parser.add_argument('--max_train_size', default=5000, type=int, help='maximum training set size')
parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default='rel_prelearning_argsort', 
    type=str, help='W&B project name')
args = parser.parse_args()

utils.print_section("SET UP")

print(f'received the following arguments: {args}')

# check if GPU is being used
print(tf.config.list_physical_devices())
assert len(tf.config.list_physical_devices('GPU')) > 0

# set up W&B logging
import wandb
wandb.login()

import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

wandb_project_name = args.wandb_project_name


def create_callbacks(monitor='loss'):
    callbacks = [
#         tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'),
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto', restore_best_weights=True))

    return callbacks

from transformer_modules import TeacherForcingAccuracy
teacher_forcing_acc_metric = TeacherForcingAccuracy(ignore_class=None)
metrics = [teacher_forcing_acc_metric]


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': args.n_epochs, 'batch_size': 128}

#region Dataset

eval_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

objects, seqs, sorted_seqs, object_seqs, target, labels, start_token = (eval_task_data['objects'], eval_task_data['seqs'], eval_task_data['sorted_seqs'], eval_task_data['object_seqs'], \
    eval_task_data['target'], eval_task_data['labels'], eval_task_data['start_token'])

num_objects, object_dim = objects.shape

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, target_train, target_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, target, labels, test_size=test_size, random_state=seed)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, target_train, target_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, target_train, labels_train, test_size=val_size/(1-test_size), random_state=seed)

seqs_length = seqs.shape[1]

source_train, source_val, source_test = object_seqs_train, object_seqs_val, object_seqs_test
#endregion

# region evaluation code

max_train_size = args.max_train_size
train_size_step = args.train_size_step
min_train_size = args.min_train_size
train_sizes = np.arange(min_train_size, max_train_size+1, step=train_size_step)

num_trials = args.num_trials # num of trials per train set size
start_trial = args.start_trial

print(f'will evaluate learning curve for `train_sizes` from {min_train_size} to {max_train_size} in increments of {train_size_step}.')
print(f'will run {num_trials} trials for each of the {len(train_sizes)} training set sizes for a total of {num_trials * len(train_sizes)} trials')

def evaluate_learning_curves(create_model, group_name, 
    source_train=source_train, target_train=target_train, labels_train=labels_train,
    source_val=source_val, target_val=target_val, labels_val=labels_val,
    source_test=source_test, target_test=target_test, labels_test=labels_test,
    train_sizes=train_sizes, num_trials=num_trials):

    for train_size in tqdm(train_sizes, desc='train size'):

        for trial in trange(start_trial, start_trial + num_trials, desc='trial', leave=False):
            run = wandb.init(project=wandb_project_name, group=group_name, name=f'train size = {train_size}; trial = {trial}',
                            config={'train size': train_size, 'trial': trial, 'group': group_name})
            model = create_model()

            sample_idx = np.random.choice(len(source_train), train_size, replace=False)
            X_train = source_train[sample_idx], target_train[sample_idx]
            y_train = labels_train[sample_idx]
            X_val = source_val, target_val
            y_val = labels_val

            history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)

            # if fitting pre-trained model, unfreeze all weights and re-train after initial training
            if args.training_mode in ['use-rel-model', 'use-rel-model-decoder']:
                stage1_epochs = max(history.epoch)
                fit_kwargs_ = {'epochs': fit_kwargs['epochs'] + max(history.epoch) + 1,
                'batch_size': fit_kwargs['batch_size'], 'initial_epoch': max(history.epoch) + 1}
                for layer in model.layers:
                    layer.trainable = True
                history = model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs_)
                stage2_epochs = max(history.epoch) - stage1_epochs
                wandb.log({'stage1_epochs': stage1_epochs, 'stage2_epochs': stage2_epochs}) # log # of epochs trained in each stage

            eval_dict = evaluate_seq2seq_model(model, source_test, target_test, labels_test, start_token, print_=False)
            log_to_wandb(model, eval_dict)
            wandb.finish(quiet=True)

            del model

# endregion


# region define models and model set up code
group_name = args.training_mode

# configuration of argsort models
argsort_model_kwargs = dict(embedding_dim=rel_model_kwargs['embedding_dim'], 
    seqs_length=seqs_length, decoder_kwargs=decoder_kwargs)

if args.training_mode == 'end-to-end':
    def create_model():
        argsort_model = AutoregressiveSimpleAbstractor(**argsort_model_kwargs)
        
        argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        argsort_model((source_train[:32], target_train[:32]));
        
        return argsort_model


elif args.training_mode == 'use-rel-model':
    def create_model():
        argsort_model = AutoregressiveSimpleAbstractor(**argsort_model_kwargs)
        
        argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        argsort_model((source_train[:32], target_train[:32]));
        
        rel_model = reload_rel_model(weights_path='prelearning_models/task1_rel_model.h5', object_dim=object_dim, kwargs=rel_model_kwargs)
        initialize_with_rel_model(argsort_model, rel_model)

        # argsort_model.abstractor.trainable = False
    
        return argsort_model

elif args.training_mode == 'use-rel-model-decoder':
    def create_model():
        argsort_model = AutoregressiveSimpleAbstractor(**argsort_model_kwargs)
        
        argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        argsort_model((source_train[:32], target_train[:32]));
        
        argsort_model_task2 = reload_argsort_model('prelearning_models/task2_argsort_model.h5', object_dim, seqs_length, argsort_model_kwargs)
        argsort_model.set_weights(argsort_model_task2.weights)

        rel_model_task1 = reload_rel_model(weights_path='prelearning_models/task1_rel_model.h5', object_dim=object_dim, kwargs=rel_model_kwargs)
        initialize_with_rel_model(argsort_model, rel_model_task1)

        return argsort_model

# endregion


# region Evaluate Learning Curves

utils.print_section("EVALUATING LEARNING CURVES")
evaluate_learning_curves(create_model, group_name=group_name)

# endregion
