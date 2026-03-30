---
layout: distill
title: What is FlashAttention?
description: A breakdown of the foundational algorithm for computing scaled dot-product attention efficiently on modern hardware
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

$$\newcommand{\matx}[1]{{\boldsymbol{#1}}}$$

I will break down FlashAttention<d-cite key="dao2022flashattention"></d-cite> in simple and easy to digest steps.

If you're new to FlashAttention, I think the best way to understand it is to think of the attention mechanism itself backwards. I go into [some detail about attention and how it is analogous to a database]({% post_url 2026-03-01-transformer %}), but I'll review the relevant bits here.



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
My notations $p$ and $r$ for the dimensions of $K$ and $V$ are called `kdim` and `vdim` in PyTorch, respectively. My $d$ is called $d_\text{model}$ in the "Attention is all you need" paper and `d_model` in PyTorch.
</aside>

with the following code is problematic:

```python
def scaled_dot_product_attention(Q, K, V):
  d = Q.shape[-1]

  s = 1. / torch.sqrt(d)
  S = s * Q @ K.T
  A = torch.nn.functional.softmax(S)
  O = A @ V

  return O
```

And the reason for that is simply because of hardware considerations. Trying to run this code on GPU will read and write matrices from and to the GPU main memory (called high-bandwith memory, or HBM), per each line of the code. Why? Because processors are too fast for the HBM memory and need a memory with a faster interface. This is called SRAM, which where the matrices go to be processed before being placed back on HBM. The catch is that SRAM is much smaller than HBM so there is no way around this dilemma. We must deal with this back and forth somehow.

This wouldn't be an issue if this was a fundamental limitation with no way around it (we'd just have to accept that bottleneck). However, linear algebra is extremely kind to us and (a) the processor doesn't have to have an entire copy of the matrices to do this, and (b) the computation doesn't need to be completed in steps, as suggested by the formulas.
<aside>
Regarding (b), the backward pass does require intermediate computations, but FlashAttention simply recomputes the necessary values when the right inputs are in SRAM. This wastes compute but the time it takes to do this is significantly shorter than the time to transfer data between HBM and SRAM so it's a price worth paying.
</aside>

## What exactly is attention trying to achieve?

Now, let's step back for a moment to try and understand what the attention mechanism is trying to do in the first place. Then, we can think about how to break down this computation into small chunks that fit in SRAM, so that the entire end-to-end computation can be done when the data is in SRAM.

As discussed in [a previous post]({% post_url 2026-03-01-transformer %}), attention can be interpreted as having the ultimate goal of finding a weighted average of a library of available "values" in $\matx{V}$. For a single query $\matx{Q}_i$, 


<div class="row mt-3 mb-5">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid
        loading="eager"
        path="assets/img/attention_value_library.png"
        class="img-fluid z-depth-1"
        caption="Attention can be understood as a weighted sum of a library of values (rainbow matrix). This figure shows the final step of attention for a single query. We are making a dish with the ingredients in the values matrix. The weights (left) indicate how much of each ingredient we must take. Mixing up (summing) the ingredients in the correct proportions (attention weights) gives the attention output (right). The attention weights matrix (left) has been transposed to be vertical for a more intuitive visualization."
        %}
    </div>
</div>

<!-- 
$$
\begin{align*}

\end{align*}
$$ -->

## How do we do this faster?

We can reduce the number of read/writes from and to HBM by performing the end-to-end computation in blocks (what is called "tiling" in the paper). In linear algebra, the elements in the matrix resulting from the product of two input matrices are simply the dot product of the corresponding rows and columns in the inputs. This means the result can be computed in steps, each time focusing only on a small "area" of the output.

## FlashAttention

