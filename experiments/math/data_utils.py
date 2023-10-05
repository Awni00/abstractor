import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

# load .env env variables (specified TFDS_DATA_DIR)
from dotenv import load_dotenv
load_dotenv();


max_q_length, max_a_length = 160, 30

start_char = '@'
eos_char = ';'

vocab = np.loadtxt('text_vectorizer_vocabs/global/vocabulary.txt', dtype=str)

q_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_q_length
)

a_text_vectorizer = tf.keras.layers.TextVectorization(
    standardize=None,
    split='character',
    output_mode='int',
    output_sequence_length=max_a_length+2
)

q_text_vectorizer.load_assets('text_vectorizer_vocabs/global')
a_text_vectorizer.load_assets('text_vectorizer_vocabs/global')

def load_dataset(task, train_size, batch_size,
    q_text_vectorizer=q_text_vectorizer, a_text_vectorizer=a_text_vectorizer,
    max_q_length=max_q_length, max_a_length=max_a_length):

    train_examples, val_examples = tfds.load(
        f'math_dataset/{task}',
        split=['train', 'test'],
        as_supervised=True)

    source_len = max_q_length
    target_len = max_a_length + 1 # max length + 2 (for start token and end token) - 1 ([:-1])
    label_len = max_a_length + 1 # max length + 2 (for start token and end token) - 1 ([1:])

    def prepend_start_token(q,a):
        source = q
        a = start_char + a + eos_char
        return q, a

    def vectorize_qa(q,a):
        return q_text_vectorizer(q), a_text_vectorizer(a)

    def get_source_target_label(q,a):
        source = q
        target = a[:-1]
        label = a[1:]
        source = tf.ensure_shape(source, (source_len,))
        target = tf.ensure_shape(target, (target_len,))
        label = tf.ensure_shape(label, (label_len,))

        return (source, target), label

    train_examples = train_examples.map(prepend_start_token).map(vectorize_qa).map(get_source_target_label)
    val_examples = val_examples.map(prepend_start_token).map(vectorize_qa).map(get_source_target_label)

    buffer_size = 16_000
    train_ds = train_examples.shuffle(buffer_size).take(train_size).cache().batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_examples.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return train_ds, val_ds

def invert_seq_vector(sequence, vectorizer):
    vocab = np.array(vectorizer.get_vocabulary())
    seq = list(vocab[sequence])
    seq = ''.join(seq)
    return seq
