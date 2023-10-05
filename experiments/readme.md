# Experiments

This directory contains code for reproducing the experiments in the paper. It contains the following experiments:

- `pairwise_order`: learn an asymmetric $\prec$ relation on randomly-generated objects.
- `object_argsort_autoregressive`: sort sequences of objects according to an underlying $\prec$ relation which needs to be learned end-to-end.
- `sorting_w_relation_prelearning`: sort sequences of randomly-generated objects by modularly learning the relation on a pair-wise subtask.
- `robustness_object_sorting`: evaluates robustness to different kinds of corruption of the object representations.
- `set`: compares the Abstractor to CoRelNet on the SET task. Also compares the relational representations produced by an Abstractor against a "symbolic" MLP which is given the relations directly (rather than having to learn them).
- `math`: evaluating abstractor architectures on mathematical problem-solving tasks.

Each directory contains a `readme` which describes the experiment in more detail and contains instructions for how to reproduce the results reported in the paper.

For all experiments, you can replicate our python environment by using the `conda_environment.yml` file, via:
```
conda env create -f conda_environment.yml
```

Many experiments have their complete logs available publicly on the ``Weights and Biases'' platform. For each run/trial, this includes metrics tracked over the course of training, the version of the code (git commit ID), the exact script that produced the run and its arguments, the hardware on which it was run, etc... See the readme associated with each experiment for the link to that experiment's logs.