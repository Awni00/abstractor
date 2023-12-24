---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: home
---

<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-MML-AM_CHTML"></script>
<!-- <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script> -->

<!-- css for buttons -->
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" />
<style>
.material-symbols-outlined {
  font-variation-settings:
  'FILL' 0,
  'wght' 400,
  'GRAD' 0,
  'opsz' 24
}
</style>
<!-- <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css"> -->
<style>
/* Style buttons */
.btn {
    background-color: DodgerBlue; /* Blue background */
    border: none; /* Remove borders */
    color: white; /* White text */
    padding: 12px 16px; /* Some padding */
    font-size: 16px; /* Set a font size */
    cursor: pointer; /* Mouse pointer on hover */
    border-radius: 5px; /* Add border radius */
    display: flex; /* Enable flex layout */
    align-items: center; /* Center vertically */
    justify-content: center; /* Center horizontally */
    ext-decoration: none;
}

/* Darker background on mouse-over */
.btn:hover {
    background-color: RoyalBlue;
    text-decoration: none;
    color: white;
}

.btn:visited {
    color: white;
}

/* Center buttons */
.button-container {
    display: flex;
    justify-content: center;
    gap: 10px;
}
</style>

<div style="text-align: center">
<h1> Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers </h1>

Awni Altabaa<sup>1</sup>, Taylor Webb<sup>2</sup>, Jonathan Cohen<sup>3</sup>, John Lafferty<sup>4</sup>
<br>
<sup>1</sup> Department of Statistics and Data Science, Yale University <br>
<sup>2</sup> Department of Psychology, UCLA <br>
<sup>3</sup> Department of Psychology and Princeton Neuroscience Institute, Princeton University <br>
<sup>4</sup> Department of Statistics and Data Science, Wu Tsai Institute, Institute for Foundations of Data Science, Yale University
</div>

<br>


<div class="button-container">
    <a href="https://arxiv.org/abs/2304.00195" class="btn" target="_blank">
    <span class="material-symbols-outlined">description</span>&nbsp;Paper&nbsp;
    </a>
    <a href="https://github.com/awni00/abstractor/" class="btn" target="_blank">
    <span class="material-symbols-outlined">code</span>&nbsp;Code&nbsp;
    </a>
    <a href="#experiment-logs" class="btn">
    <span class="material-symbols-outlined">experiment</span>&nbsp;Experimental Logs&nbsp;
    </a>
</div>

<br>

<figure style="text-align: center;">
    <!-- <img src="figs/....png" alt="..."> -->
    <figcaption>Figure: A depiction of .</figcaption>
</figure>

## Abstract


## Method Overview

...

Please see the paper for more details on the proposed architecture.


## Experiments

We evaluate our proposed architecture on... . Please see the paper for a description of the tasks and the experimental set up. We include a preview of the results here.


## Experiment Logs

Detailed experimental logs are publicly available. They include training and validation metrics tracked during training, test metrics after training, code/git state, resource utilization, etc.


## Citation

```
@article{altabaa2023abstractors,
      title={Abstractors and relational cross-attention: An inductive bias for explicit relational reasoning in Transformers}, 
      author={Awni Altabaa and Taylor Webb and Jonathan Cohen and John Lafferty},
      year={2023},
      eprint={2304.00195},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```