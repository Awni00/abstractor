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
    <img src="figs/self_attn_fig.png" alt="..." style="width: 49%; display: inline-block;">
    <img src="figs/rel_crossattn_fig.png" alt="..." style="width: 49%; display: inline-block;">
    <figcaption style="text-align: left;">Figure: A depiction of relational cross-attention (right), compared with standard self-attention (left). Relational cross-attention implements a type of information bottleneck to disentangle object-level features from relational features. When integrated into a broader Transformer-based architecture, this enables explicit relational representation, yielding improved abstraction and generalization from limited data.</figcaption>
</figure>

## Abstract

An extension of Transformers is proposed that enables explicit relational reasoning through a novel module called the *Abstractor*. At the core of the Abstractor is a variant of attention called *relational cross-attention*. The approach is motivated by an architectural inductive bias for relational learning that disentangles relational information from extraneous features about individual objects. This enables explicit relational reasoning, supporting abstraction and generalization from limited data. The Abstractor is first evaluated on simple discriminative relational tasks and compared to existing relational architectures. Next, the Abstractor is evaluated on purely relational sequence-to-sequence tasks, where dramatic improvements are seen in sample efficiency compared to standard Transformers. Finally, Abstractors are evaluated on a collection of tasks based on mathematical problem solving, where modest but consistent improvements in performance and sample efficiency are observed.

## Method Overview

The core operation in a Transformer is attention. For an input sequence $$X = (x_1, \ldots, x_n)$$, self-attention transforms the sequence via, $$ X' \gets \phi_v(X) \, \mathrm{Softmax}({\phi_q(X)}^\top \phi_k(X))$$, where $$\phi_q, \phi_k, \phi_v$$ are functions on $$\mathcal{X}$$ applied independently to each object in the sequence (i.e., $$\phi(X) = (\phi(x_1), \ldots, \phi(x_n))$$).
We think of $$R := \phi_q(X)^\top \phi_k(X)$$ as a relation matrix. Self-attention admits an interpretation as a form of neural message-passing as follows, 

$$x_i' \gets \mathrm{MessagePassing}\left(\{(\phi_v(x_j), R_{ij})\}_{j \in [n]}\right) = \sum_{j} \bar{R}_{ij} \phi_v(x_j).$$

where $$m_{j \to i} = (\phi_v(x_j), R_{ij})$$ is the message from object $$j$$ to object $$i$$, encoding the sender's features and the relation between the two objects, and $$\bar{R} = \mathrm{Softmax}(R)$$ is the softmax-normalized relation matrix. Hence, the processed representation obtained by self-attention is an entangled mixture of relational information and object-level features.

Our goal is to learn relational representations which are abstracted away from object-level features in order to achieve more sample-efficient learning and improved generalization in relational reasoning. This is not naturally supported by the entangled representations produced by standard self-attention. We achieve this via a simple modification of attention---we replace the values $$\phi_v(x_i)$$ with vectors that *identify* objects, but do not encode any information about their features. We call those vectors *symbols*. Hence, the message sent from object $$j$$ to object $$i$$ is now $$m_{j \to i} = (s_j, R_{ij})$$, the relation between the two objects, together with with the symbol identifying the sender object,

$$A_i \gets \mathrm{MessagePassing}\left(\{(s_j, R_{ij})\}_{j \in [n]}\right)$$

Symbols act as abstract references to objects. They do not contain any information about the contents or features of the objects, but rather *refer* to objects.

Abstraction relies on the assignment of symbols to individual objects without directly encoding their features. We propose three different mechanisms for assigning symbols to objects: *positional symbols* (symbols are assigned to objects sequentially based on the order they appear), *position-relative symbols* (symbols are assigned to encode the relative position of the sender with respect ot the receiver), and *symbolic attention* (symbols are retrieved via attention from a library of learned symbols).

This motivates a variant of attention which we call *relational cross-attention*. Relational cross-attention forms the core of the *Abstractor* module. The Abstractor processes a sequence of objects to produce a sequence of "abstract embeddings" which encode the relational features among its input.

Please see the paper for the details.

## Experiments

We evaluate our proposed architecture on... . Please see the paper for a description of the tasks and the experimental set up. We include a preview of the results here.

[TODO]

## Experiment Logs

Detailed experimental logs are publicly available. They include training and validation metrics tracked during training, test metrics after training, code/git state, resource utilization, etc.


[TODO]

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