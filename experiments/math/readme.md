# Mathematical problem-solving

This set of experiments evaluates Abstractor architectures on a set of mathematical problem-solving tasks from the [`mathematics_dataset`](https://github.com/google-deepmind/mathematics_dataset) contributed by Saxton, Grefenstette, Hill, and Kohli.

Steps to reproduce experiments:

For each `task`, `model`, `model_size`, `trial`, etc., run the following:
```
python train_model.py --model {model} --task {task} --model_size {model_size} --run_name trial={trial} --n_epochs 50 --batch_size 128 --train_size -1
```

In our paper, we report experiments on the following `task`'s: `['calculus__differentiate', 'algebra__sequence_next_term', 'algebra__linear_1d', 'polynomials__expand', 'polynomials__add']`. The code supports other tasks in the `mathematics_dataset` by specifying the task name. We evaluate the following `model`'s: `['relational_abstractor', 'transformer']`. For the Abstractor, we evaluate a `model_size` of `'medium'`. For the Transformer, we evaluate `model_size`'s `['medium', 'medium+']`.

The complete logs of our experiments are available at the links below:

| Task                        	| Experimental logs                                        	|
|-----------------------------	|----------------------------------------------------------	|
| calculus__differentiate     	| https://wandb.ai/awni00/math-calculus__differentiate     	|
| algebra__sequence_next_term 	| https://wandb.ai/awni00/math-algebra__sequence_next_term 	|
| algebra__linear_1d          	| https://wandb.ai/awni00/math-algebra__linear_1d          	|
| polynomials__expand         	| https://wandb.ai/awni00/math-polynomials__expand         	|
| polynomials__add            	| https://wandb.ai/awni00/math-polynomials__add            	|