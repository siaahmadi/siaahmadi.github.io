---
layout: distill
title: The softmax function and its gradient
description: Detailed derivation of the gradient of softmax
tags: ai, optimization, neural-network, math
categories: AI, from-scratch
giscus_comments: false
date: 2026-03-05
featured: false
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
    url: "https://siaahmadi.github.io/from_scratch"
    affiliations:
      name: Independent Researcher

bibliography: 2026-03-05-softmax-gradient.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: The softmax function
  - name: The softmax Jacobian
  - name: The gradient when softmax is an intermediate computation
    subsections:
      - name: Masking operation

# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
_styles: >
  .fake-img {
    background: #bbb;
    border: 1px solid rgba(0, 0, 0, 0.1);
    box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
    margin-bottom: 12px;
  }
  .fake-img p {
    font-family: monospace;
    color: white;
    text-align: left;
    margin: 12px 0;
    text-align: center;
  }
    font-size: 16px;
---

$$
\newcommand{\smx}[1]{\mathrm{softmax}(#1)}
\newcommand{\Vector}[1]{\mathbf{#1}}
\newcommand{\D}[1]{\frac{d}{d#1}}
\newcommand{\Dp}[2][]{\frac{\partial#1}{\partial#2}}
\newcommand{\smxv}[1]{\mathrm{softmax}(\Vector{#1})}
$$

# The softmax function

$$
\begin{align*}
    \Vector{x} & = 
    \begin{bmatrix}
        x_1 \\
        x_2 \\
        \vdots \\
        x_n
    \end{bmatrix} \in \mathbb{R}^n\\[3mm]
    \smx{\Vector{x}} & =
    \begin{bmatrix}
        \frac{e^{x_1}}{\sum_ke^{x_k}} & \frac{e^{x_2}}{\sum_ke^{x_k}} & \ldots & \frac{e^{x_n}}{\sum_ke^{x_k}}
    \end{bmatrix} \\[1mm]
    \smxv{x}_j & = \frac{e^{x_j}}{\sum_ke^{x_k}} = \frac{u(x_j)}{v(\Vector{x})}
\end{align*}
$$

where $u(x) = e^x$ and $v(\Vector{x}) = \sum_ke^{x_k}$.

# The softmax Jacobian

Recall the quotient derivative rule:

$$
\begin{align}
        \frac{d}{dx}(\frac{u(x)}{v(x)}) = \frac{u'(x)v(x) - u(x)v'(x)}{[v(x)]^2} \label{eq:quotient_derivative}
\end{align}
$$

We have

$$
\begin{align}
    \Dp{x_i}u(x_j) & = \begin{cases}
        u(x_i) & , i = j\\ 
        0 & , i \neq j\\ 
    \end{cases} = \delta_{ij}u(x_i) \label{eq:uprime} \\[3mm]
    \Dp{x_i}v(\Vector{x}) & = u(x_i) \label{eq:vprime}
\end{align}
$$

where $\delta_{ij}=1$ when $i=j$ and 0 otherwise. Now,

$$
\begin{align*}
    \Dp{x_i}\smxv{x}_j & = \frac{[\Dp{x_i}u(x_j)]v(\Vector{x}) - u(x_j)[\Dp{x_i}v(\Vector{x})]}{[v(\Vector{x})]^2} && \tag*{(by \eqref{eq:quotient_derivative})}\\[2mm]
    & = \frac{\delta_{ij}u(x_i)v(\Vector{x}) - u(x_j)u(x_i)}{v(\Vector{x})v(\Vector{x})}  && \tag*{(by \eqref{eq:uprime} and \eqref{eq:vprime})}\\[2mm]
    & = \frac{u(x_i)}{v(\Vector{x})}\cdot\frac{\delta_{ij}v(\Vector{x}) - u(x_j)}{v(\Vector{x})}\\[2mm]
    & = \frac{u(x_i)}{v(\Vector{x})}\cdot\left(\delta_{ij}\frac{v(\Vector{x})}{v(\Vector{x})}-\frac{u(x_j)}{v(\Vector{x})}\right)\\[2mm]
    & = \frac{u(x_i)}{v(\Vector{x})}\cdot\left(\delta_{ij}-\frac{u(x_j)}{v(\Vector{x})}\right)\\[2mm]
    & = \boxed{\smxv{x}_i(\delta_{ij}-\smxv{x}_j)} = \smxv{x}_j(\delta_{ij}-\smxv{x}_i)
\end{align*}
$$


If we set $\Vector{p} = \smxv{x}$, a vectorized implementation of this could look like this:

```python
def softmax_grad(x):
    p = softmax(x, axis=-1)
    grad = np.diag(p) - p.T @ p
    return grad
```


Here's a horrendously detailed explanation of why this works:

$$
\begin{align*}
    \Vector{p} & = \begin{bmatrix}
        p_1 & p_2 & \cdots & p_n
    \end{bmatrix} \\[2mm]
    \mathrm{diag}(\Vector{p}) & = \begin{bmatrix}
        p_1 & 0 & \cdots & 0 \\
        0 & p_2 & \cdots & 0 \\
        \vdots & & & \vdots \\
        0 & 0 & \cdots & p_n \\
    \end{bmatrix} \\[4mm]
    \Vector{p}\top \Vector{p} & = \begin{bmatrix}
        p_1 \\
        p_2 \\
        \vdots\\
        p_n
    \end{bmatrix} \begin{bmatrix}
        p_1 & p_2 & \cdots & p_n
    \end{bmatrix} = \begin{bmatrix}
        p_1p_1 & p_1p_2 & \cdots & p_1p_n \\
        p_2p_1 & p_2p_2 & \cdots & p_2p_n \\
        \vdots & & & \vdots \\
        p_np_1 & p_np_2 & \cdots & p_np_n \\
    \end{bmatrix} \\[6mm]
    \mathrm{diag}(\Vector{p}) - \Vector{p}\top \Vector{p} & = \begin{bmatrix}
        p_1 & 0 & \cdots & 0 \\
        0 & p_2 & \cdots & 0 \\
        \vdots & & & \vdots \\
        0 & 0 & \cdots & p_n \\
    \end{bmatrix} - \begin{bmatrix}
        p_1p_1 & p_1p_2 & \cdots & p_1p_n \\
        p_2p_1 & p_2p_2 & \cdots & p_2p_n \\
        \vdots & & & \vdots \\
        p_np_1 & p_np_2 & \cdots & p_np_n \\
    \end{bmatrix} \\[6mm]
    & = \begin{bmatrix}
        p_1 - p_1p_1 & -p_1p_2 & \cdots & -p_1p_n \\
        -p_2p_1 & p_2 - p_2p_2 & \cdots & -p_2p_n \\
        \vdots & & & \vdots \\
        -p_np_1 & -p_np_2 & \cdots & p_n - p_np_n \\
    \end{bmatrix}\\[6mm]
    & = \begin{bmatrix}
        p_1 (1 - p_1) & p_1(0-p_2) & \cdots & p_1(0-p_n) \\
        p_2 (0 - p_1) & p_2(1-p_2) & \cdots & p_2(0-p_n) \\
        \vdots & & & \vdots \\
        p_n (0 - p_1) & p_n(0-p_2) & \cdots & p_n(1-p_n) \\
    \end{bmatrix}\\[6mm]
    & = \begin{bmatrix}
        p_1 (\delta_{ij} - p_1) & p_1(\delta_{ij}-p_2) & \cdots & p_1(\delta_{ij}-p_n) \\
        p_2 (\delta_{ij} - p_1) & p_2(\delta_{ij}-p_2) & \cdots & p_2(\delta_{ij}-p_n) \\
        \vdots & & & \vdots \\
        p_n (\delta_{ij} - p_1) & p_n(\delta_{ij}-p_2) & \cdots & p_n(\delta_{ij}-p_n) \\
    \end{bmatrix}\\[6mm]
    & = \Vector{p}_i(\delta_{ij} - \Vector{p}_j) = \nabla_{x_i}\smxv{x}_j
\end{align*}
$$


# The gradient when softmax is an intermediate computation

When we are computing the gradient of an upstream loss with respect to the softmax inputs, something interesting happens. In this case, we don't need to compute the Jacobian first, as seen below:

$$
\begin{align*}
    G & = \nabla_\Vector{p}\mathcal{L}\in\mathbb{R}^{1\times n} \qquad\qquad \text{# upstream gradient}\\[4mm]
    \Dp[\mathcal{L}]{\Vector{x}} & = \Dp[\mathcal{L}]{\Vector{p}}\cdot\Dp[\Vector{p}]{\Vector{x}} \in \mathbb{R}^{1\times n}\\[2mm]
    & = G(\mathrm{diag}(\Vector{p})-\Vector{p}^\top\Vector{p})\\[2mm]
    & = \underbrace{G\cdot\mathrm{diag}(\Vector{p})}_{\text{matrix multiply}}-\underbrace{G\cdot\underbrace{\Vector{p}^\top\Vector{p}}_{\text{matrix}}}_{\text{matrix multiply}}\\[2mm]
    & = \underbrace{G \odot \Vector{p}}_{\text{scalar products}}-\underbrace{\underbrace{(G\cdot\Vector{p}^\top)}_{\text{dot product}}\Vector{p}}_{\text{scalar by vector}}\\[2mm]
    & \Rightarrow \boxed{\nabla_\Vector{x}\mathcal{L} = (G - G\cdot\Vector{p})\odot\Vector{p}}\in\mathbb{R}^{1\times n}
\end{align*}
$$

<aside>
<p>
$(p^\top)_{n\times1}\Vector{p}_{1\times n}=P_{n\times n}$ is a matrix. $G_{1\times n}$ itself is a vector, so $G\Vector{p}^\top\Vector{p}$ is a vector-matrix multiplication. However, by matrix multiplication commutativity, we can carry out $G\Vector{p}^\top$ first which is a dot product of two vectors, resulting in a scalar. Then we multiply this scalar by the vector $\Vector{p}$.
</p>
<p>
Likewise, since we're using $\mathrm{diag}(\cdot)$ to denote a diagonal matrix whose diagonal elements are given by the vector argument, $G\mathrm{diag}(\Vector{p})$ is also a vector-by-matrix multiplication. This is the same as elementwise multiplication of $G$ and $\Vector{p}$.
</p>
<p>
Once we write the equations, we can factor out a $\Vector{p}$ from the right, which gives the final equation.
</p>
</aside>

So now the code becomes

```python
def softmax_grad(x, g):
    p = softmax(x, axis=-1)
    g_dot_p = np.vecdot(g, p)[..., np.newaxis]
    grad    = (g - g_dot_p) * p
    return grad
```

## Masking operation

Causal masking of the softmax inputs (that is, setting the upper triangular part of the matrix to $-\infty$) will not affect the gradient calculations. This is because the $\Vector{p}$ term in the final formula will simply contain 0s where we have $-\infty$, meaning gradients won't flow to those entries. This is also intuitive as nudging around the $-\infty$'s by scalars will leave them at $-\infty$ and not change the probabilities produce by the finite logits, meaning the gradient with respect to the masked values should be 0.