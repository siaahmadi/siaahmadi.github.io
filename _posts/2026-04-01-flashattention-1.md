---
layout: distill
title: What is FlashAttention?
description: Easy-to-digest breakdown of the foundational algorithm for computing scaled dot-product attention efficiently on modern hardware
tags:
  - systems
  - optimization
  - transformer
  - attention
categories:
  - from-scratch
  - AI
giscus_comments: true
date: 2025-04-01
featured: true
mermaid:
  enabled: true
  zoomable: true
code_diff: true
map: true
chart:
  chartjs: true
  echarts: true
  vega_lite: true
tikzjax: true
typograms: true

authors:
  - name: Siavash Ahmadi
    url: "https://siaahmadi.github.io/"
    affiliations:
      name: Independent Researcher

bibliography: 2026-04-01-flashattention-1.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: Equations
    # if a section has subsections, you can add them as follows:
    # subsections:
    #   - name: Example Child Subsection 1
    #   - name: Example Child Subsection 2
  - name: Citations

---

$$
\newcommand{\matx}[1]{\boldsymbol{#1}}
$$



If you're new to FlashAttention<d-cite key="dao2022flashattention"></d-cite>, I think the best way to understand it is to think of the attention mechanism itself backwards. I go into [some detail about attention and how it is analogous to a database]({% post_url 2026-03-01-transformer %}), but I'll review the relevant bits here.



## What's wrong with standard attention?

Before, I go into that I want to make sure I mentioned why computing scaled dot-product attention give by the formula

$$
\begin{align*}
\matx{Q}\in\mathbb{R}^{N\times d} &, \matx{K}\in\mathbb{R}^{p\times d}, \matx{V}\in\mathbb{R}^{p\times r}\\
s & = d^{-\frac{1}{2}}\\
\matx{S} & = s\matx{Q}\matx{K}^\top\\
\matx{A} & = \mathrm{softmax}(\matx{S})\\
\matx{O} & = \matx{A}\matx{V} \in \mathbb{R}^{N\times r}
\end{align*}
$$
<aside>
My notations $p$ and $r$ for the dimensions of $K$ and $V$ are called `kdim` and `vdim` in PyTorch, respectively. My $d$ is called $d_\text{model}$ in the "Attention is all you need" paper and `d_model` in PyTorch. For multihead attention, $d = d_\text{model} / h$ where $h$ is the number of attention heads.
</aside>

with the following code is problematic:

```python
def scaled_dot_product_attention(Q, K, V):
  d = Q.shape[-1]

  s = 1. / torch.sqrt(d)
  S = s * Q @ K.T
  A = torch.nn.functional.softmax(S, dim=-1)
  O = A @ V

  return O
```

And the reason for that is simply because of hardware considerations. Trying to run this code on GPU will read and write matrices from and to the GPU main memory (called high-bandwith memory, or HBM), per each line of the code. Why? Because processors are too fast for the HBM memory and need a memory with a faster interface. This is called SRAM, which where the matrices go to be processed before being placed back on HBM. The catch is that SRAM is much smaller than HBM so there is no way around this dilemma. We must deal with this back and forth somehow.

This wouldn't be an issue if this was a fundamental limitation with no way around it (we'd just have to accept that bottleneck). However, linear algebra is extremely kind to us and (a) the processor doesn't have to have an entire copy of the matrices to do this, and (b) the computation doesn't need to be completed in staged steps, as suggested by the formulas.
<aside>
Regarding (b), the backward pass does require intermediate computations, but FlashAttention simply recomputes the necessary values when the right inputs are in SRAM. This wastes compute but the time it takes to do this is significantly shorter than the time to transfer data between HBM and SRAM so it's a price worth paying.
</aside>

## What exactly is attention trying to achieve?

Now, let's step back for a moment to try and understand what the attention mechanism is trying to do in the first place. Then, we can think about how to break down this computation into small chunks that fit in SRAM, so that the entire end-to-end computation can be done when the data is in SRAM.

As discussed in [a previous post]({% post_url 2026-03-01-transformer %}), attention can be interpreted as having the ultimate goal of finding a weighted average of a library of available "values" in $\matx{V}$. Considering the attention weights $\matx{A}_i$ resulting from a single query $\matx{Q}_i$ ($i$ indexes the rows here),

$$
\begin{align*}
  \matx{O}_i = \matx{A}_i\matx{V} = \sum_k \matx{A}_{ik}\matx{V}_k
\end{align*}
$$

represents the final vector we'd like to get out. Figure 1 shows this pictorially.

<div class="row mt-3">
    <div class="col-sm z-depth-1 p-3">
        <figure class="text-center mb-0">
          {%
            include figure.liquid
            loading="eager"
            path="assets/img/attention_value_library.png"
            class="img-fluid"
          %}
          <figcaption class="caption mt-3">
              <b>Figure 1</b>. Attention can be understood as a weighted sum of a library of values (rainbow matrix). This figure shows the final step of attention for a single query. We are making a dish with the ingredients in the values matrix. The weights (left) indicate how much of each ingredient we must take. Mixing up (summing) the ingredients in the correct proportions (attention weights) gives the attention output (right). In this figure, I have transposed the attention weights matrix (left) into a column for a more intuitive visualization.
          </figcaption>
        </figure>
    </div>
</div>


## How do we do this faster?

We can reduce the number of read/writes from and to HBM by performing the end-to-end computation in blocks (what is called "tiling" in the paper). In linear algebra, the elements in the matrix resulting from the product of two input matrices are simply the dot product of the corresponding rows and columns in the inputs. This means the result can be computed in steps, each time focusing only on a small "area" of the output.

But to simplify everything, suppose that the computation shown in Figure 1 was all we wanted to do. Then it should be clear that this can be performed in "chunks" as Figure 2 shows:




<div class="row mt-3">
    <div class="col-sm z-depth-1 p-3 bg-white">
        <figure class="text-center mb-0">
            <p class="text-left fw-bold fs-4 my-4"> Chunk 1: </p>
            {% include figure.liquid loading="eager" path="assets/img/flashattention-dot-product-step-1.png" class="img-fluid mb-2" %}
            <p class="text-left fw-bold fs-4 my-4"> Chunk 2: </p>
            {% include figure.liquid loading="eager" path="assets/img/flashattention-dot-product-step-2.png" class="img-fluid mb-2" %}
            <p class="text-left fw-bold fs-4 my-4"> Chunk 3: </p>
            {% include figure.liquid loading="eager" path="assets/img/flashattention-dot-product-step-3.png" class="img-fluid" %}
            <figcaption class="caption mt-3">
                <b>Figure 2</b>. Final step of the attention mechanism, performed in chunks (or blocks, or tiles).
            </figcaption>
        </figure>
    </div>
</div>



## FlashAttention

At its heart, FlashAttention consists of three separate aspects:

1. The core computation
2. Normalization factor
3. Numerical stability

The core computation is what I call matmul-exp-dot-product, which captures the end-to-end computation by multiplying $\matx{Q}$ by $\matx{K}^\top$, exponentiating (softmax numerator), and taking the dot product of the result with the library of value vectors in $\matx{V}$.

The second operation is necessary to ensure proper normalization of the weighted averaging discussed in Figure 1.

The third is the familiar idea that we don't want to exponentiate large values in the computation of [the softmax function]({% post_url 2026-03-05-softmax-gradient %}), so we first subtract the largest value from all input elements before exponentiating.

I will strip away (2) and (3) and focus on the matmul-exp-dot-product to show how this is done end-to-end without moving data between HBM and SRAM more than necessary.

### The core computation of attention

Consider the $i$-th row of the query matrix, $\matx{Q}_i$. The attention output for this query is

$$
\begin{align}
  \matx{O}_i =
  \begin{bmatrix}
    \displaystyle\sum_{k=1}^p\frac{\exp(\matx{Q}_i\matx{K}_1^\top)}{\ell_i}\matx{V}_{k1} &
    \displaystyle\sum_{k=1}^p\frac{\exp(\matx{Q}_i\matx{K}_2^\top)}{\ell_i}\matx{V}_{k2} &
    \cdots &
    \displaystyle\sum_{k=1}^p\frac{\exp(\matx{Q}_i\matx{K}_p^\top)}{\ell_i}\matx{V}_{kr}
  \end{bmatrix}
\end{align}
$$

  <!-- % \\
  % \begin{bmatrix}
  %   \displaystyle\frac{\exp(\matx{Q}_i\matx{K}_1^\top)}{\ell_i} &
  %   \displaystyle\frac{\exp(\matx{Q}_i\matx{K}_2^\top)}{\ell_i} &
  %   \cdots &
  %   \displaystyle\frac{\exp(\matx{Q}_i\matx{K}_p^\top)}{\ell_i}
  % \end{bmatrix}\matx{V}
  % \\
  % \begin{bmatrix}
  %   \matx{Q}_i\matx{K}_1^\top &
  %   \matx{Q}_i\matx{K}_2^\top &
  %   \cdots &
  %   \matx{Q}_i\matx{K}_p^\top
  % \end{bmatrix}
  % \\
  % \matx{Q}_i
  % \begin{bmatrix}
  %   \matx{K}_1^\top & \matx{K}_2^\top & \cdots & \matx{K}_p^\top
  % \end{bmatrix} -->

where

$$
\ell_i = \sum_{k=1}^p\exp(\matx{Q}_i\matx{K}_k^\top).
$$

Ignoring the denominator $\ell_i$ (aspect (2) in the above list) for a moment, we'll have the following barebones equation

$$
\begin{align} \label{eq:barebones}
  \tilde{\matx{O}}_i =
  \begin{bmatrix}
    \displaystyle\sum_{k=1}^p\exp(\matx{Q}_i\matx{K}_1^\top)\matx{V}_{k1} &
    \displaystyle\sum_{k=1}^p\exp(\matx{Q}_i\matx{K}_2^\top)\matx{V}_{k2} &
    \cdots &
    \displaystyle\sum_{k=1}^p\exp(\matx{Q}_i\matx{K}_p^\top)\matx{V}_{kr}
  \end{bmatrix}
\end{align}
$$

and now it is clear that this equation can be calculated for any $k\leq p$ iteratively (meaning we don't have to have any of the values for $j\neq k$ ready yet). Figure 2 illustrated the dot product step of this equation.

The matmul-exp-dot-product operation I mentioned earlier should be clearly identifiable now: the "matmul" is the product inside the exponential, the "exp" is the exponential, and the "dot-product" is the sum over the products with columns of $\matx{V}$. Figure 3 shows the matmul-exp step.


<div class="row mt-3">
    <div class="col-sm z-depth-1 p-3 bg-white">
        <figure class="text-center mb-0">
            {% include figure.liquid loading="eager" path="assets/img/flashattention-matmulexp.png" class="img-fluid mb-2" %}
            <figcaption class="caption mt-3">
                <b>Figure 3</b>. Matmul-exp operations in the barebones attention \eqref{eq:barebones}. We have chosen a block from $K$ (its third and fourth rows) to compute the attention scores for a single query. The "Attention Weights" shows the logical representation of the final result, by indicating the previously computed values (first two) as small circles (in HBM), the values just computed (next two) as pentograms (in SRAM), and yet-uncomputed values (last three) as squares.
            </figcaption>
        </figure>{ : #fig-matmulexp }
    </div>
</div>



> ##### KEY POINT
>
> At its core, **FlashAttention** iteratively computes Eq. $\eqref{eq:barebones}$ by selecting a block of rows from $\matx{K}$ and $\matx{V}$ (say from row $n_1$ to row $n_2$).
{: .block-tip }

Figure 4 below shows this graphically.

<div class="row mt-3">
    <div class="col-sm z-depth-1 p-3 bg-white">
        <figure class="text-center mb-0">
            {% include figure.liquid loading="eager" path="assets/img/flashattention-barebones.png" class="img-fluid mb-2" %}
            <figcaption class="caption mt-3">
                <b>Figure 4</b>. End-to-end computation of barebones matmul-exp-dot-product attention \eqref{eq:barebones}. The computation is iterative, so the values accumulate, which is evident in the penultimate step (notice the already-computed grayed out values which we recall from HBM into SRAM for accumulation). Note that because we are ignoring the normalization factor (Aspect 2 of FlashAttention), the final vector is "fatter" than it should be (for comparison, Figure 2 performs the normalization).
            </figcaption>
        </figure>
    </div>
</div>
