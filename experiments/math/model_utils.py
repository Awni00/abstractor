import tensorflow as tf
import numpy as np
import wandb

from data_utils import a_text_vectorizer, start_char

import sys; sys.path.append('../..')
from transformer_modules import TeacherForcingAccuracy
from multi_head_attention import MultiHeadAttention

custom_objects = dict(TeacherForcingAccuracy=TeacherForcingAccuracy, MultiHeadAttention=MultiHeadAttention)

def autoregressive_predict(model, source, target):
    n, seqs_length = np.shape(target)

    output = np.zeros(shape=(n, (seqs_length+1)), dtype=int)
    output[:,0] = a_text_vectorizer.get_vocabulary().index(start_char)

    for i in range(seqs_length):
        predictions = model((source, output[:, :-1]), training=False)
        predictions = predictions[:, i, :]
        predicted_id = np.argmax(predictions, axis=-1)
        output[:,i+1] = predicted_id

    return output[:, 1:]

def fetch_model(artifact_path, artifact_root_dir='model_artifacts'):
    api = wandb.Api()
    artifact = api.artifact(artifact_path, type='model')
    artifact_dir = artifact.download(f"{artifact_root_dir}/{artifact_path.replace('/','-')}")
    model = tf.keras.models.load_model(artifact_dir, custom_objects=custom_objects)
    return model

def recompile_model(model, a_text_vectorizer):
    ignore_class = a_text_vectorizer.get_vocabulary().index('')
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, ignore_class=None)
    teacher_forcing_accuracy = TeacherForcingAccuracy(ignore_class=ignore_class)
    model.compile(loss=loss, optimizer='adam', metrics=teacher_forcing_accuracy)
    return model