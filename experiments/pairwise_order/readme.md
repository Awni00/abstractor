# Pairwise order relation

This simple experiment tests the Abstractors ability to learn asymmetric relations. Similar to `object_argsort_autoregressive`, we generate $N = 64$ different random objects $`\mathcal{O} = \{o_i\}_{i\in[N]}`$ and associate a strict ordering relation to them. The input to the model is a pair of two objects $(o_i, o_j)$ and the task is to predict whether $o_i \prec o_j$. Of the $N^2 = 64$ tuples, we split it in 50% training, 15% validation (used for early stopping), and 35% testing. Hence, the models will have never seen the input pair they receive at evaluation time and need to generalize based on the transitivity of $\prec$. We evaluate learning curves for each model.

Steps to replicate results:
1) Run `pairwise_order_relation_learning_curves.ipynb`. This generates the synthetic dataset, and evaluates learning curves.