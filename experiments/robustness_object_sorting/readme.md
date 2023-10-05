# Robustness Experiments
Consider the same object-sorting task in `object_sorting_autoregressive`. In this experiment, we evaluate robustness to different kind of noise. In particular, we train an Abstractor, Transformer, and an "ablation model" on the task to saturation. Then, we corrupt the objects in $\mathcal{O}$ in different ways and evaluate the models on the sequences in the hold-out test set. We evaluate the following kinds of corruption.

Additive noise:
$$\tilde{o}_i = o_i + \varepsilon_i, \quad \text{where } \varepsilon_i \sim \mathcal{N}(0, \sigma^2 I).$$

Random linear transformation:

$$\tilde{o}_i = A o_i, \quad \text{where } A_{ij} \overset{iid}{\sim} \mathcal{N}(0, \sigma^2).$$

Random orthogonl transformation:
$$\tilde{o}_i = A o_i, \quad \text{where } A = (1 - \alpha) I + \alpha B,\ B \sim \text{Unif}(O(d)).$$

Steps to reproduce results in paper:

1) Run `generate_random_object_sorting_tasks.ipynb` to generate the dataset(s).

2) Run the following to train each model on the task and save the model weights
    ```
    python train_models.py --n_epochs 400 --early_stopping False --train_size 3000 --wandb_project_name robustness_object_sorting
    ```

3) Run `robustness_evaluation.ipyb` to evaluate the three types of robustness described above.

The logs for step 2 are available to view at: [W&B project link](https://wandb.ai/awni00/robustness_object_sorting).