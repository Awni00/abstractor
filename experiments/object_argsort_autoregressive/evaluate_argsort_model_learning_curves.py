# RANDOM OBJECT AUTOREGRESSIVE ARGSORT 
# We generate random objects (as gaussian vectors) and associate an ordering to them.
# We train abstractor models to learn how to sort these objects
# To test the generalization of abstracters, we first train one on another object-sorting task, 
# then initialize w those weights and train on the primary task
# The models do 'argsorting', meaning they predict the argsort of the sequnce.

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import argparse

import tensorflow as tf

import sklearn.metrics
from sklearn.model_selection import train_test_split

import sys; sys.path.append('../'); sys.path.append('../..')
import seq2seq_abstracter_models
import autoregressive_abstractor
import utils
from eval_utils import evaluate_seq2seq_model, log_to_wandb

# region SETUP

seed = None

# parse script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,
    choices=('transformer', 'abstractor', 'rel-abstracter', 'sym-abstracter', 'simple-abstractor', 'ablation-abstractor'),
    help='the model to evaluate learning curves on')
parser.add_argument('--pretraining_mode', default='none', type=str,
    choices=('none', 'pretraining'),
    help='whether and how to pre-train on pre-training task')
parser.add_argument('--init_trainable', default=True, type=bool,
    help='whether or not to make initialized weights trainable when pre-trainign (in first stage)')
parser.add_argument('--pretraining_task_type', default='independent objects', 
    type=str, choices=('NA', 'independent objects', 'reshuffled objects', 'reshuffled attr'))
parser.add_argument('--pretraining_task_data_path', default='object_sorting_datasets/task1_object_sort_dataset.npy', 
    type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--eval_task_data_path', default='object_sorting_datasets/task2_object_sort_dataset.npy', 
    type=str, help='path to npy file containing sorting task dataset')
parser.add_argument('--pretraining_train_size', default=1_000, type=int,
    help='training set size for pre-training (only used for pre-training tasks)')
parser.add_argument('--n_epochs', default=500, type=int, help='number of epochs to train each model for')
parser.add_argument('--early_stopping', default=True, type=bool, help='whether to use early stopping')
parser.add_argument('--min_train_size', default=500, type=int, help='minimum training set size')
parser.add_argument('--max_train_size', default=5000, type=int, help='maximum training set size')
parser.add_argument('--train_size_step', default=50, type=int, help='training set step size')
parser.add_argument('--num_trials', default=1, type=int, help='number of trials per training set size')
parser.add_argument('--start_trial', default=0, type=int, help='what to call first trial')
parser.add_argument('--wandb_project_name', default='abstractor_object_argsort', 
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


def create_callbacks(monitor='val_teacher_forcing_accuracy'):
    callbacks = [
#         tf.keras.callbacks.ReduceLROnPlateau( monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto'),
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor=monitor, patience=50, mode='max', restore_best_weights=True))

    return callbacks

from transformer_modules import TeacherForcingAccuracy
teacher_forcing_acc_metric = TeacherForcingAccuracy(ignore_class=None)
metrics = [teacher_forcing_acc_metric]


loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None, name='sparse_categorical_crossentropy')
create_opt = lambda : tf.keras.optimizers.Adam()

fit_kwargs = {'epochs': args.n_epochs, 'batch_size': 512}

#region Dataset

eval_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

objects, seqs, sorted_seqs, object_seqs, target, labels, start_token = (eval_task_data['objects'], eval_task_data['seqs'], eval_task_data['sorted_seqs'], eval_task_data['object_seqs'], \
    eval_task_data['target'], eval_task_data['labels'], eval_task_data['start_token'])

test_size = 0.2
val_size = 0.1

seqs_train, seqs_test, sorted_seqs_train, sorted_seqs_test, object_seqs_train, object_seqs_test, target_train, target_test, labels_train, labels_test = train_test_split(
    seqs, sorted_seqs, object_seqs, target, labels, test_size=test_size, random_state=seed)
seqs_train, seqs_val, sorted_seqs_train, sorted_seqs_val, object_seqs_train, object_seqs_val, target_train, target_val, labels_train, labels_val = train_test_split(
    seqs_train, sorted_seqs_train, object_seqs_train, target_train, labels_train, test_size=val_size/(1-test_size), random_state=seed)

seqs_length = seqs.shape[1]

source_train, source_val, source_test = object_seqs_train, object_seqs_val, object_seqs_test
#endregion

# region kwargs for all the models
transformer_kwargs = dict(
    num_layers=4, num_heads=2, dff=64, 
    input_vocab='vector', target_vocab=seqs_length+1,
    output_dim=seqs_length, embedding_dim=64)

rel_abstractor_kwargs = dict(
    num_layers=2, num_heads=2, dff=64, 
    input_vocab='vector', target_vocab=seqs_length+1,
    output_dim=seqs_length, embedding_dim=64,
    rel_attention_activation='softmax'
    )

simple_abstractor_kwargs = dict(
    embedding_dim=64, 
    input_vocab='vector', target_vocab=seqs_length+1, output_dim=seqs_length,
    abstractor_kwargs=dict(num_layers=1, num_heads=4, dff=64,
        use_pos_embedding=False, mha_activation_type='softmax'),
    decoder_kwargs=dict(num_layers=1, num_heads=4, dff=64, dropout_rate=0.1))

ablation_abstractor_kwargs = dict(
    num_layers=2, num_heads=2, dff=64, 
    input_vocab='vector', target_vocab=seqs_length+1,
    output_dim=seqs_length, embedding_dim=64,
    use_self_attn=True, use_encoder=True,
    mha_activation_type='softmax'
    )

autoreg_abstractor_kwargs = dict(
        encoder_kwargs=dict(num_layers=2, num_heads=4, dff=64, dropout_rate=0.1),
        abstractor_kwargs=dict(
            num_layers=2,
            rel_dim=4,
            symbol_dim=64,
            proj_dim=8,
            symmetric_rels=False,
            encoder_kwargs=dict(use_bias=True),
            rel_activation_type='softmax',
            use_self_attn=False,
            use_layer_norm=False,
            dropout_rate=0.2),
        decoder_kwargs=dict(num_layers=1, num_heads=4, dff=64, dropout_rate=0.1),
        input_vocab='vector',
        target_vocab=seqs_length+1,
        embedding_dim=64,
        output_dim=seqs_length,
        abstractor_type='abstractor',
        abstractor_on='encoder',
        decoder_on='abstractor',
        name='autoregressive_abstractor')

# endregion

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
            if 'pretraining' in args.pretraining_mode and not args.init_trainable:
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

# pre-training set up
if 'pretraining' in args.pretraining_mode:

    # load pre-training task data
    pretraining_task_data = np.load(args.eval_task_data_path, allow_pickle=True).item()

    object_seqs_pretraining, target_pretraining, labels_pretraining, start_token_pretraining = (pretraining_task_data['object_seqs'], \
        pretraining_task_data['target'], pretraining_task_data['labels'], pretraining_task_data['start_token'])

    test_size = 0.2
    val_size = 0.1

    (object_seqs_train_pretraining, object_seqs_test_pretraining, target_train_pretraining, target_test_pretraining, 
        labels_train_pretraining, labels_test_pretraining) = train_test_split(
        object_seqs_pretraining, target_pretraining, labels_pretraining, test_size=test_size, random_state=seed)
    (object_seqs_train_pretraining, object_seqs_val_pretraining, target_train_pretraining, target_val_pretraining, 
        labels_train_pretraining, labels_val_pretraining) = train_test_split(
        object_seqs_pretraining, target_pretraining, labels_pretraining, test_size=val_size/(1-test_size), random_state=seed)

    (source_train_pretraining, source_val_pretraining, source_test_pretraining) = (object_seqs_train_pretraining,
        object_seqs_val_pretraining, object_seqs_test_pretraining)

    X_train = (source_train_pretraining[:args.pretraining_train_size], target_train_pretraining[:args.pretraining_train_size])
    y_train = labels_train_pretraining[:args.pretraining_train_size]
    X_val = (source_val_pretraining[:args.pretraining_train_size], target_val_pretraining[:args.pretraining_train_size])
    y_val = labels_val_pretraining[:args.pretraining_train_size]

    

# transformer
if args.model == 'transformer':
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.Transformer(
                **transformer_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Transformer'
    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = seq2seq_abstracter_models.Transformer(
            **transformer_kwargs)

        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        pretrained_model((source_train_pretraining[:32], target_train_pretraining[:32]));

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group=f'Pre-training Task ({args.pretraining_task_type}); Transformer', 
            config={
                'train size': args.pretraining_train_size, 
                'group': f'Pre-training Task ({args.pretraining_task_type}); Transformer',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_seq2seq_model(pretrained_model, source_test_pretraining, target_test_pretraining, labels_test_pretraining,
            start_token_pretraining,  print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':
            def create_model():
                argsort_model = seq2seq_abstracter_models.Transformer(
                    **transformer_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                # TODO: think about whether initializing / freezing encoder weights makes sense for transformer
                # argsort_model.encoder.set_weights(pretrained_model.encoder.weights)
                # argsort_model.encoder.trainable = False

                argsort_model.decoder.set_weights(pretrained_model.decoder.weights)
                argsort_model.decoder.trainable = args.init_trainable

                return argsort_model

            group_name = f'Transformer (Pre-Trained; {args.pretraining_task_type})'

    else:
        raise NotImplementedError(f'`pretraining_mode` {args.pretraining_mode} is invalid`')

# Autoregressive Abstractor
elif args.model == 'abstractor':
    # standard evaluation
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = autoregressive_abstractor.AutoregressiveAbstractor(
                **autoreg_abstractor_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Abstractor'
    
    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = autoregressive_abstractor.AutoregressiveAbstractor(
            **autoreg_abstractor_kwargs)

        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        pretrained_model((source_train_pretraining[:32], target_train_pretraining[:32]));

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group=f'Pre-training Task ({args.pretraining_task_type}); Abstractor', 
            config={
                'train size': args.pretraining_train_size, 
                'group': f'Pre-training Task ({args.pretraining_task_type}); Abstractor',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_seq2seq_model(pretrained_model, source_test_pretraining, target_test_pretraining, labels_test_pretraining,
            start_token_pretraining,  print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':
            def create_model():
                argsort_model = autoregressive_abstractor.AutoregressiveAbstractor(
                **autoreg_abstractor_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                for model_layer, pretrained_layer in zip(argsort_model.layers[:-1], pretrained_model.layers[:-1]):
                    model_layer.set_weights(pretrained_layer.weights)
                    model_layer.trainable = args.init_trainable

                return argsort_model

            group_name = f'Abstractor (Pre-Trained; {args.pretraining_task_type})'
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

# Relational abstracter
elif args.model == 'rel-abstracter':
    # standard evaluation
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.Seq2SeqRelationalAbstracter(
                **rel_abstractor_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Relational Abstractor'
    
    # TODO: think about how to initialize / pre-train
    # perhaps need to initialize everything (including embedders, etc)

    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = seq2seq_abstracter_models.Seq2SeqRelationalAbstracter(
            **rel_abstractor_kwargs)

        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        pretrained_model((source_train_pretraining[:32], target_train_pretraining[:32]));

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group=f'Pre-training Task ({args.pretraining_task_type}); Relational Abstractor', 
            config={
                'train size': args.pretraining_train_size, 
                'group': f'Pre-training Task ({args.pretraining_task_type}); Relational Abstractor',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_seq2seq_model(pretrained_model, source_test_pretraining, target_test_pretraining, labels_test_pretraining,
            start_token_pretraining,  print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':
            def create_model():
                argsort_model = seq2seq_abstracter_models.Seq2SeqRelationalAbstracter(
                    **rel_abstractor_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                argsort_model.abstracter.set_weights(pretrained_model.abstracter.weights)
                argsort_model.abstracter.trainable = args.init_trainable

                argsort_model.decoder.set_weights(pretrained_model.decoder.weights)
                argsort_model.decoder.trainable = args.init_trainable

                return argsort_model

            group_name = f'Relational Abstractor (Pre-Trained; {args.pretraining_task_type})'
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

# Simple abstracter
elif args.model == 'simple-abstractor':
    # standard evaluation
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.AutoregressiveSimpleAbstractor(
                **simple_abstractor_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Simple Abstractor'
    
    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = seq2seq_abstracter_models.AutoregressiveSimpleAbstractor(
            **simple_abstractor_kwargs)

        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        pretrained_model((source_train_pretraining[:32], target_train_pretraining[:32]));

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group=f'Pre-training Task ({args.pretraining_task_type}); Relational Abstractor', 
            config={
                'train size': args.pretraining_train_size, 
                'group': f'Pre-training Task ({args.pretraining_task_type}); Simple Abstractor',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_seq2seq_model(pretrained_model, source_test_pretraining, target_test_pretraining, labels_test_pretraining,
            start_token_pretraining,  print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':
            def create_model():
                argsort_model = seq2seq_abstracter_models.AutoregressiveSimpleAbstractor(
                    **simple_abstractor_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                argsort_model.set_weights(pretrained_model.weights)
                
                argsort_model.source_embedder.trainable = args.init_trainable
                argsort_model.abstractor.trainable = args.init_trainable
                argsort_model.decoder.trainable = args.init_trainable
                # what remains is the final dense layer. this is always trainable

                return argsort_model

            group_name = f'Simple Abstractor (Pre-Trained; {args.pretraining_task_type})'
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

# Symbolic abstracter
elif args.model == 'sym-abstracter':
    # standard evaluation
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.Seq2SeqSymbolicAbstracter(
                **rel_abstractor_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Symbolic Abstractor'
    
    # if evaluating generalization via pre-training
    elif args.pretraining_mode in ['pretraining']:
        
        # fit model on pre-training task
        pretrained_model = seq2seq_abstracter_models.Seq2SeqSymbolicAbstracter(
            **rel_abstractor_kwargs)

        pretrained_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
        pretrained_model((source_train_pretraining[:32], target_train_pretraining[:32]));

        utils.print_section('Fitting Model on Pre-training Task')
        run = wandb.init(project=wandb_project_name, name=f'pretraining_mode={args.pretraining_mode}',
            group=f'Pre-training Task ({args.pretraining_task_type}); Symbolic Abstractor', 
            config={
                'train size': args.pretraining_train_size, 
                'group': f'Pre-training Task ({args.pretraining_task_type}); Symbolic Abstractor',
                'pretraining_mode': args.pretraining_mode}
            )
        history = pretrained_model.fit(X_train, y_train, validation_data=(X_val, y_val), verbose=0, callbacks=create_callbacks(), **fit_kwargs)
        eval_dict = evaluate_seq2seq_model(pretrained_model, source_test_pretraining, target_test_pretraining, labels_test_pretraining,
            start_token_pretraining,  print_=False)
        log_to_wandb(pretrained_model, eval_dict)
        wandb.finish(quiet=True)

        if args.pretraining_mode == 'pretraining':
            def create_model():
                argsort_model = seq2seq_abstracter_models.Seq2SeqSymbolicAbstracter(
                    **rel_abstractor_kwargs)

                argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
                argsort_model((source_train[:32], target_train[:32]));

                argsort_model.abstracter.set_weights(pretrained_model.abstracter.weights)
                argsort_model.abstracter.trainable = args.init_trainable

                argsort_model.decoder.set_weights(pretrained_model.decoder.weights)
                argsort_model.decoder.trainable = args.init_trainable

                return argsort_model

            group_name = f'Symbolic Abstractor (Pre-Trained; {args.pretraining_task_type})'
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

elif args.model == 'ablation-abstractor':
    # standard evaluation
    if args.pretraining_mode == 'none':
        def create_model():
            argsort_model = seq2seq_abstracter_models.AutoregressiveAblationAbstractor(
                **ablation_abstractor_kwargs)

            argsort_model.compile(loss=loss, optimizer=create_opt(), metrics=metrics)
            argsort_model((source_train[:32], target_train[:32]));

            return argsort_model
        
        group_name = 'Ablation Abstractor'
    
    else:
        raise ValueError(f'`pretraining_mode` {args.pretraining_mode} is invalid')

else:
    raise ValueError(f'`model` argument {args.model} is invalid')


# endregion


# region Evaluate Learning Curves

utils.print_section("EVALUATING LEARNING CURVES")
evaluate_learning_curves(create_model, group_name=group_name)

# endregion
