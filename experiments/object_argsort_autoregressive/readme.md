# Sorting Random Objects

Consider a set of randomly generated objects $`\mathcal{O} = \{ o_1, ..., o_N \}`$, where each $o_i$ is sampled iid from some distribution. We associate a strict ordering relation to these objects $o_1 \prec o_2 \prec \cdots \prec o_N$. Given a randomly permuted sequence of objects, the task is to predict the argsort.

Steps to reproduce experiments:
1) Run `generate_random_object_sorting_tasks.ipynb` to generate the random objects and the sorting tasks. This generates a dataset containing $(x,y)$ tuples where the input $x$ is a random permutation of objects and $y$ is the argsort of the objects according to $\prec$.
2) Run `evaluate_argsort_model_learning_curves.py` with the desired parameters to evaluate learning curves for a particular problem. To replicate the results in the paper, run the following for each `model_name` in `['rel-abstracter', 'simple-abstractor', 'transformer', 'ablation-abstractor']`:


    ```
    python evaluate_argsort_model_learning_curves.py --model {{model_name}} --pretraining_mode "none" --eval_task_data_path "object_sorting_datasets/product_structure_object_sort_dataset.npy" --n_epochs 400 --early_stopping True --min_train_size 100 --max_train_size 3000 --train_size_step 100 --num_trials 10 --start_trial 0 --pretraining_train_size 1000 --wandb_project_name "object_argsort_autoregressive"
    ```

    This replicates the results in section 4.2. To replicate the results in section 4.3, run the following for each `model_name` in `['simple-abstractor', 'transformer']`:

    ```
    python evaluate_argsort_model_learning_curves.py --model {{model_name}} --pretraining_mode "pretraining" --init_trainable True --pretraining_task_type "reshuffled attr" --pretraining_task_data_path object_sorting_datasets/product_structure_reshuffled_object_sort_dataset.npy --eval_task_data_path object_sorting_datasets/product_structure_object_sort_dataset.npy --n_epochs 400 --early_stopping True --min_train_size 100 --max_train_size 3000 --train_size_step 100 --num_trials 10 --start_trial 0 --pretraining_train_size 3000 --wandb_project_name "object_argsort_autoregressive"
    ```

3) Plotting and analyzing learning curves is done in `learning_curve_analysis.ipynb`

Our experiments used the parameters above. The complete logs of our runs are available to view at: [W&B project link](https://wandb.ai/awni00/object_argsort_autoregressive?workspace=user-awni00).