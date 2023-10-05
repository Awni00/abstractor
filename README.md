# Abstractors

This is the repository associated with the paper
> "Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers" --- Awni Altabaa, Taylor Webb, Jonathan D. Cohen, John Lafferty.

The arXiv version is here: https://arxiv.org/abs/2304.00195.

The following is an outline of the repo:

- `abstracters.py` and `abstractor.py` implement different variants of the Abstractor module.
- `autoregressive_abstractor.py` implements sequence-to-sequence abstractor-based architectures. `seq2seq_abstracter_models.py` is an older, less general, implementation of sequence-to-sequence models.
- `multi_head_attention.py` is a fork of tensorflow's implementation which we have adjusted to support different kinds of activation functions applied to the attention scores. `transformer_modules.py` includes implementations of different Transformer modules (e.g.: Encoders, Decoders, etc.). Finally, `attention.py` implements different attention mechanisms for Transformers and Abstractors (including relational cross-attention).
- The `experiments` directory contains the code for all experiments in the paper. See the `readme`'s therein for details on the experiments and instructions for replicating them.
- The `paper` directory contains the source for the paper itself.