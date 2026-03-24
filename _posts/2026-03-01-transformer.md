---
layout: distill
title: The Transformer
description: Discussions of the Transformer architecture, atlernate visualizations, and reinterpretation as a database.
tags: ai, optimization, neural-network, math, transformer
categories: AI, from-scratch
giscus_comments: false
date: 2026-03-01
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
    url: "https://siaahmadi.github.io/from_scratch"
    affiliations:
      name: Independent Researcher

bibliography: 2026-03-01-transformer.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: The Transformer Architecture
    subsections:
      - name: Transformer architecture, better visualized
      - name: Attention as a generalization of a database
  - name: Full Derivation of Multihead Attention
    subsections:
      - name: 

---


# The Transformer Architecture

I've had to read [the main Transformer paper](https://doi.org/10.48550/arXiv.1706.03762) many times in the past to fully understand it. And from time to time there seemed to be something that escaped my mind and I had to go back to it again. Each time I was a bit confused while reading the paper and I thought the paper could have been written better and more clearly. As conceptually simple as it seems (and it is simple really) the paper is pretty dense actually, and the figures are pretty bad in my view. I'm going to try and explain it here again, in a way that seems more straightforward and intuitive to myself. Hopefully, you you find it useful.

## Transformer architecture, better visualized

The figures in the Transformer paper are really confusing to me. I think just because they are technically "correct" it doesn't mean they communicate the ideas well. For example, there is no figure that puts together the entire attention mechanism end-to-end in a simplified way. Below are my several attempts at doing so:

<center><img src="./images/attention_layout1.png" alt="End-to-end attention schematic for a single head" title="Attention mechanism, end-to-end" width="800"/></center>

This is the original paper's description of the transformer self-attention. Note that it's *self-attention* not *cross-attention* because the inputs are all the same (the "self" is the input giving rise to $K$ and $V$). This schematic is meant to illustrate a simplified picture and show you how the different matrix sizes match up. The boxes are sized carefully to match the matrix product dimension rules. For each matrix multiplication $C=AB$ the matrix $A_{m\times p}$ is situation to the left, matrix $B_{p\times n}$ is situated to the top and $C_{m\times n}$ is placed at the center. If you think of these boxes in 3D, we're "lifting up" $A$ and $B$ and requiring that their heights $p$ be the same. Of course the length $m$ and widht $n$ can vary arbitrarily.

Nvidia has a really intuitive illustration of this to demonstrate how their tensor cores work:


<center>
<div style="width: 650px; overflow: hidden;">
  <img src="https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tensorcore/nvidia-tensor-cores-og-social-1200x630-r1.jpg" 
       style="width: 1200px; max-width: none; height: auto; display: block;"
       title="Intuititive visualization of matrix multiplication by Nvidia"
       alt="Matrix multiplication"
       >
</div>
</center>

(Image displayed directly from the source: [https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tensorcore/nvidia-tensor-cores-og-social-1200x630-r1.jpg](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tensorcore/nvidia-tensor-cores-og-social-1200x630-r1.jpg); if viewing on Github, chances are this is not cropping properly so please just focus on the illustration on the left)

Here, the cyan matrix is $A$, the purple is $B$, and the results $C$ is in green. The gray 3D strucutre in the middle is the result of the multplications before the sumation (recall $C_{ij}=\sum_kA_{ik}B_{kj}$, so each vertical column of the gray structure is indexed by $k$ and comes from the product of the corresponding row $i$ in $A$ and column $j$ in $B$). My schematic is this picture viewed from above, and with the cyan and purple matrices laid flat.

Here's the same schematic, rearranged, in case you find it easier to comprehend this way:

<center><img src="./images/attention_layout2.png" alt="End-to-end attention schematic for a single head" title="Attention mechanism, end-to-end (alternative layout)" width="600"/></center>

But if you're really used to neural networks being represented as multi-layer perceptrons, then the following might be the best representation for you. This simultaneously shows a) the whole attention mechanism end to end, b) the context-dependent nature of attention, which is reminiscient of meta-learning, and c) that the key `K` and value `V` have a special meaning in attention (hence, KV caching, why `K` and `V` can come from an encoder layer, why they might have different dimensionalities in multi-modal models, etc.).

<center><img src="./images/attention_layout3.png" alt="nd-to-end attention schematic for a single head revealing its context dependence and the unique significance K and V matrices" title="Attention mechanism, end-to-end (meta learning layout)" width="800"/></center>

The key here is that we can choose $q$, $k$ and $v$ inputs according to out needs. For example, in a decoder-only these are $q = k = v = x$, leading to what we call "self-attention." Of course $x$ refers to the $d_{\mathrm{model}}$-dimensional word embeddings in the initial layer and to the "information highway" (of the same dimensionality) thereafter.

In an encoder-decoder layout, the $k$ and $v$ would come from the encoder, and the $q$ comes from the decoder's "information highway." This is what we call "cross-attention." The $k$ and $v$ can also come from an image embedding encoder for a multi-modal model. This is why the sequence lengths ($L_q$ vs $L_k$) can differ between these depending on the embedder's chosen dimensionality.

An important point here is that PyTorch's impelmentation of the transformer layers does not allow the user to choose $d_k$ and $d_v$ (these are always set to $d_{\mathrm{model}}$ / num_heads). In this package, however, this code does allow control over all of those parameters.

# Attention as a generalization of a database

(I can't recall where I first encountered this interpretation but it may have been by Andrej Karpathy.)

The core of the transformer is the attention mechanism (forget about the bells and whistels like "multi-head" and the "output" and the "dense layer" and "layer norm" etc etc; even the "in-projection" steps should be ignored for now). At the most basic level the transformer is powerful because it does a context-dependent computation (unlike, say, a [multi-layer perceptron]({% post_url 2026-02-18-mlp %})).

Below I'm comparing the attention mechanism to a Python dictionary. I use the terms "dictionary" and "database" interchangeably.

In Python, we can do this:

```
db = {’a’: 1, ‘b’: 2}
query = 'a'
print(db[query]) # prints 1
query = 'c'
print(db[query]) # error: key not in dictionary
```

In essence, here's what's happening under the hood. `db` contains a set of `key:value` pairs. When we call `db[query]`, `query` is "compared against every `key` in `db`" (in quotes because this is not the algorithm that actually runs under the hood, but it _can_ be thought of in these terms). If `query` matches any key (`if exists`) then the `value` associated with the matching `key` is returned. Otherwise, this means that the `query` does not exists as a `key` in `db` and `error` is raised.

What we're *ultimately* interested in in a database lookup is the *values* we get--the query and key are simply a means to that end. In the transformer attention mechanism, we are simply looking to extract a weighted average of all of our (context-dependent) values `V`.

The transformer attention generalizes this "hard" lookup with a "soft" lookup. The `query` is given as `Q`. The "dictionary" always contains some set of keys `K` (these may be useless, as in an untrained network, or useful, as in a well-trained network). When we `Q @ K.T` in the scaled dot product, we are doing the "compare against every key" step: the matrix multiply simultaneously performs a bunch of dot products. And what is a dot product geometrically? A *similarity metric*!

```
A = Q . K --> similarity: -1 if opposite directions, 0 if orthogonal, +1 if the same
```

Because this computation is done using real numbers and calculus, we are in the *continuous* regime (unlike the discrete/binary regime of the Python dictionary). Furthermore, the entries in `Q`, `K`, and `V` are high-dimensional vectors. This means the query `Q` is always "similar" (that's to say, a quantity can always be computed) to all of the keys in `K` (of which we have `L`), but the ***degree of similarity*** can be low or high. The application of the softmax function on top of this "lookup" operation is a convenience for making sure we get an appropriately weighted average (i.e., weights sum to 1) of the values `V`.

```
Q in R^{1 x D} # A single D-dimensional object
K in R^{L x D} # L entries in the dictionary (each D-dimensions)

Q @ K^T --> A in R^{1 x L} # similarity of the query to each and every entry in the database
```

The dictionary lookup reduces to the attention mechanism if in the attention algorithm we used binary values. For example, let's say we have compared our `query` in `Q` to all of the `key`'s in `K` and observed that it is the same as the fourth element. We have this information in `A` below:
```
A = [0, 0, 0, 1, 0], V in R^{5 x D_v}
O = A @ V in R^{1 x D_v}
```

In the second row above, note that a weighted sum when `A` is a one-hot encoded vector is the same as just selecting the ith vector in `V`! So we have done the same as:

```
db = {
    1: d_dimensional_vector_1,
    2: d_dimensional_vector_2,
    3: d_dimensional_vector_3,
    4: d_dimensional_vector_4,
    5: d_dimensional_vector_5,
}
O = db[4] # returns `d_dimensional_vector_4`
```




<!-- 
\newcommand{\smx}[1]{\mathrm{softmax}(#1)}
\newcommand{\Vector}[1]{\mathbf{#1}}
\newcommand{\D}[1]{\frac{d}{d#1}}
\newcommand{\Dp}[2][]{\frac{\partial#1}{\partial#2}}
\newcommand{\smxv}[1]{\mathrm{softmax}(\Vector{#1})}
\newcommand{\W}[1]{\Vector{W^{#1}}}
\newcommand{\Wq}[0]{\Vector{W^q}}
\newcommand{\Wqi}[0]{\Vector{W}_i^\Vector{q}}
\newcommand{\Wk}[0]{\Vector{W^k}}
\newcommand{\Wki}[0]{\Vector{W}_i^\Vector{k}}
\newcommand{\Wv}[0]{\Vector{W^v}}
\newcommand{\Wvi}[0]{\Vector{W}_i^\Vector{v}}
\newcommand{\Wo}[0]{\Vector{W^o}}
\newcommand{\Aa}[0]{\Vector{a}}
\newcommand{\Z}[0]{\Vector{Z}}
\newcommand{\Q}[0]{\Vector{Q}}
\newcommand{\K}[0]{\Vector{K}}
\newcommand{\V}[0]{\Vector{V}}
\newcommand{\A}[0]{\Vector{A}}
\newcommand{\Ll}[0]{\mathcal{L}}

# Full Derivation of Multihead Attention


Below, I'll be repeating the same idea multiple times, progressively adding bells and whistles to hopefully make this easier to follow. The calculations are presented in tensor (N-D array) format to make them directly compatible with NumPy, with $N$ referring to the number of batches.

## Attention

NOTE: Don't be intimidated by attention! If right off the bat you want to see a single equation that fully describes multihead attention, check out Eq. \eqref{eqn:multihead_attn}. Also see Figure \ref{fig:mha}!
The core of the Transformer architecture is the attention function, which is surprisingly simple and straightforward to describe. Attention is a function of three inputs arguments (which are matrices) and one output argument (which is a matrix), and can be separated into three semantically distinct steps:


* In-projections
* Scaled dot-product attention
* Out-projection


Note that these three steps are just names we're using to partition the operations of the attention function in an easier to digest way. Attention is just the composition of several function. These the names refer to the following equations:



* In-projections
    \begin{align*}
        \Vector{Q} = \Vector{q}\Vector{W^q}, \quad \Vector{K} = \Vector{k}\Vector{W^k}, \quad \Vector{V} = \Vector{v}\Vector{W^v}
    \end{align*}
    Here, the $\Vector{W}$ superscripts little $^\Vector{q}$, little $^\Vector{k}$, and little $^\Vector{v}$ are not indexes---they are just part of the variable's name.
* Scaled dot-product attention
    \begin{align*}
        \Vector{A} = \smx{\frac{\Vector{Q}\Vector{K}^\top}{\sqrt{d_k}}}\Vector{V}\\
    \end{align*}
    where $d_k$ is a scalar hyperparameter we will get to later.
* Out-projection
    \begin{align*}
        \Vector{Z} = \Vector{A}\Vector{W^o}
    \end{align*}
    Here, too, the superscript little $^\Vector{o}$ is part of the variable's name and is not an index.


## Multiheaded Attention

The transformer uses multiheaded attention which is just a fancy-sounding way to say it repeats the above operations multiple times using different values for the four $\Vector{W}$ parameter matrices.

The transformer multiheaded attention boils down to the following equations, where we now use the $i=1,\ldots, h$ subscript \emph{indexes} to distinguish between the different \emph{heads} and write the whole thing a bit more formally:

$$
\begin{eqnarray}
    \Vector{Q}_i = \Vector{q}\Vector{W}_i^\Vector{^q}, \quad \Vector{K}_i = \Vector{k}\Vector{W}_i^\Vector{^k}, \quad \Vector{V}_i = \Vector{v}\Vector{W}_i^\Vector{^v}\\
    \Vector{A}_i = \smx{\frac{\Vector{Q}_i\Vector{K}_i^\top}{\sqrt{d_k}}}\Vector{V}_i\\
    \Vector{Z} = \begin{bmatrix}
        \Vector{A}_1 & \Vector{A}_2 & \cdots & \Vector{A}_h
    \end{bmatrix}\Vector{W^o}
\end{eqnarray}
$$

Putting it all together we get the following (multiheaded attention in a single equation):

$$
\begin{align}
    \Vector{Z}(\Vector{q}, \Vector{k}, \Vector{v}) = \begin{bmatrix}
        \displaystyle\smx{\frac{\Vector{q}\Vector{W}_1^\Vector{^q}(\Vector{k}\Vector{W}_1^\Vector{^k})^\top}{\sqrt{d_k}}}\Vector{v}\Vector{W}_1^\Vector{^v}\\[6mm]
        \displaystyle\smx{\frac{\Vector{q}\Vector{W}_2^\Vector{^q}(\Vector{k}\Vector{W}_2^\Vector{^k})^\top}{\sqrt{d_k}}}\Vector{v}\Vector{W}_2^\Vector{^v}\\[6mm]
        \vdots\\[6mm]
        \displaystyle\smx{\frac{\Vector{q}\Vector{W}_h^\Vector{^q}(\Vector{k}\Vector{W}_h^\Vector{^k})^\top}{\sqrt{d_k}}}\Vector{v}\Vector{W}_h^\Vector{^v}\\[6mm]
    \end{bmatrix}^\top\Vector{W^o} \label{eqn:multihead_attn}
\end{align}
$$

where the three-input $\Vector{Z}$ (the multiheaded attention with $h$ heads) is parameterized by $3h+1$ matrices: $\Vector{W}_i\Vector{^q}$, $\Vector{W}_i\Vector{^k}$, $\Vector{W}_i\Vector{^v}$, and $\Vector{W^o}$, $i=1,\ldots,h$.\\

The ``hardest'' thing about implementing multiheaded attention from scratch is managing the matrix shapes effectively so as to make the matrix multiplications efficient.

## Full description of multiheaded attention operations

### In projections



\begin{center}
\begin{tabular}{ |c|c|c|c| } 
 \hline
 Inputs & Parameters & Forward pass operations & Outputs \\[1mm]
 \hline
 & & &\\[-2mm]
 $\Vector{q} \in \mathbb{R}^{N\times L_q\times d_\text{model}}$ & $\Vector{W_q} \in \mathbb{R}^{h\times d_\text{model}\times d_k}$ & \multirow{2}{55mm}{\texttt{Q = np.dot(q,Wq)}\\\texttt{Q = np.moveaxis(Q, -2, 0)}} & $\Vector{Q} \in \mathbb{R}^{h\times N\times L_q\times d_k}$ \\
 & & & \\[5mm]
 % \hline
 $\Vector{k} \in \mathbb{R}^{N\times L_k\times kdim}$ & $\Vector{W_k} \in \mathbb{R}^{h\times kdim\times d_k}$ & \multirow{2}{55mm}{\texttt{K = np.dot(k,Wk)}\\\texttt{K = np.moveaxis(K, -2, 0)}} & $\Vector{K} \in \mathbb{R}^{h\times N\times L_k\times d_k}$ \\
 & & & \\[5mm]
 % \hline
 $\Vector{v} \in \mathbb{R}^{N\times L_k\times vdim}$ & $\Vector{W_v} \in \mathbb{R}^{h\times vdim\times d_v}$ & \multirow{2}{55mm}{\texttt{V = np.dot(v,Wv)}\\\texttt{V = np.moveaxis(V, -2, 0)}} & $\Vector{V} \in \mathbb{R}^{h\times N\times L_k\times d_v}$ \\
 & & & \\[2mm]
 \hline
\end{tabular}
\end{center}



Here, $d_\text{model} = hd_k$ is the \emph{dimensionality of the model}, i.e., the dimensionality of the token embeddings coming in. We typically choose $d_\text{model}$ and $h$ and calculate $d_k$ as $\frac{d_\text{model}}{h}$, implying we require $d_\text{model}$ to be divisible by $h$.


### Scaled dot product attention
#### Scaled dot product

\begin{center}
\begin{tabular}{ |c|c|c| } 
 \hline
 Inputs & Forward pass operations & Outputs \\
 \hline
 & &\\[-2mm]
 \multirow{3}{35mm}{$\Vector{Q} \in \mathbb{R}^{h\times N\times L_q\times d_k}$\\[3mm]$\Vector{K} \in \mathbb{R}^{h\times N\times L_k\times d_k}$} & \multirow{3}{75mm}{\texttt{s = 1 / np.sqrt(d\_k)\\S = Q @ np.swapaxes(K, -1, -2) * s}} & \multirow{3}{30mm}{$\Vector{S}\in\mathbb{R}^{h\times N\times L_q \times L_k}$} \\
  &  &   \\[6mm]
 \hline
\end{tabular}
\end{center}

#### Masked Softmax

For the softmax gradient, see \href{../utils/softmax_gradient.pdf}{softmax and its gradient (PDF)}.

\begin{center}
\begin{tabular}{ |c|c|c| } 
 \hline
 Inputs & Forward pass operations & Outputs \\[1mm]
 \hline
 & &\\[-2mm]
 $\Vector{S}\in\mathbb{R}^{h\times N\times L_q \times L_k}$ & See softmax and its gradient PDF & $\Vector{A}\in\mathbb{R}^{h\times N\times L_q \times L_k}$ \\[2mm]
 \hline
\end{tabular}
\end{center}



#### Attention-weighted values

\newcommand{\aconcat}[0]{\Aa'}

\begin{center}
\begin{tabular}{ |c|c|c| } 
 \hline
 Inputs & Forward pass operations & Outputs \\
 \hline
 \multirow{4}{35mm}{$\A\in\mathbb{R}^{h\times N\times L_q \times L_k}$\\[2mm]$\Vector{V} \in \mathbb{R}^{h\times N\times L_k\times d_v}$} & \multirow{4}{75mm}{\texttt{a = A @ V \# shape (h, N, L\_q, d\_v)\\a' = np.concatenate(a, axis=-1)}} & \multirow{4}{30mm}{$\Aa'\in\mathbb{R}^{N\times L_q \times hd_v}$} \\
 & & \\
 & & \\[4mm]
 \hline
\end{tabular}
\end{center}


\begin{figure}[!b]
    \centering
    \includegraphics[width=0.7\linewidth]{attention_layout3.pdf}
    \caption{\textbf{Attention function and the residual stream}. This figure visually describes the function discussed in this document. It also shows the residual stream (bottom arrow and the summation $\oplus$) and the (optional) dropout layer. Note that this representation obfuscates some intermediate steps which have been made explicit in the equations of this document.}
    \label{fig:mha}
\end{figure}


We used a different name $\aconcat$ for the concatenated form of $\Aa$ because later this will make reasoning about the gradients clearer.

\subsection{Out projection}

\begin{center}
\begin{tabular}{ |c|c|c|c| } 
 \hline
 Inputs & Parameters & Forward pass operations & Outputs \\[1mm]
 \hline
 & & &\\[-2mm]
 $\Vector{a}'\in\mathbb{R}^{N\times L_q \times hd_v}$ & $\Vector{W_o} \in \mathbb{R}^{hd_v\times d_\text{model}}$ & \texttt{Z = a' @ Wo} & $\Vector{Z} \in \mathbb{R}^{N\times L_q\times d_\text{model}}$ \\[2mm]
 \hline
\end{tabular}
\end{center}

Note that the output $\Vector{Z}$ has the exact same dimensionality as the input $\Vector{q}$, and this is by design, which is why we can add the residual stream and stack multiple Multiheaded Attention blocks sequentially. To see this graphically for a single head, see Figure \ref{fig:mha}.





\section{The gradients (backward pass)}

We begin deriving the gradients from the last operation, the out projection, and work our way back to the inputs. For any operation $y = f(x)$, we will use the shorthand

\[
G = \Dp[\mathcal{L}]{y} \in \mathcal{X}
\]

for the gradient of the final loss function $\mathcal{L}$ with respect to the current operation's output $y$. $\mathcal{X}$ is the set to which both $G$ and $y$ belong (determines dimensionality of both). We refer to $G$ as the ``upstream gradient.''

\subsection{Out projection}

\newcommand{\Zforward}[0]{
\Z & = \aconcat\Wo = \begin{bmatrix}
        \Aa_1 & \Aa_2 & \cdots & \Aa_h
    \end{bmatrix} \Wo \in \mathbb{R}^{N\times L_q\times d_\text{model}}
}
\newcommand{\dZda}[0]{
\Dp[\Z]{\aconcat} = \Wo^\top \in \mathbb{R}^{d_\text{model} \times hd_v}
}
\newcommand{\dZdWo}[0]{
\Dp[\Z]{\Wo} = \aconcat^\top \in \mathbb{R}^{N\times hd_v \times L_q}
}
\newcommand{\dLda}[0]{
\Dp[\Ll]{\aconcat} = G\Dp[\Z]{\aconcat} \in\mathbb{R}^{N\times L_q \times hd_v}
}
\newcommand{\dLdWo}[0]{
\Dp[\Ll]{\Wo} = \Dp[\Z]{\Wo}G \in \mathbb{R}^{N\times hd_v \times d_\text{model}}
}

\begin{align*}
    \Zforward && \tag*{(forward pass)} \\[3mm]
    \dZda & ,\quad \dZdWo && \tag*{(local gradients)}\\[3mm]
    \dLda & ,\quad \dLdWo && \tag*{(upstream of lower operations)}\\
\end{align*}

To match the dimensionality of $\Dp[\Ll]{\Wo}$ with that of $\Wo$, we must sum over the batch dimension\footnote{This is not the \emph{real} reason but it sounds like a better mnemonic. The real reason is that by performing the preceding calculations, we have computed $N$ different gradients as a result of the $N$ batches, and that, by chain rule, gradients flowing in to a parameter from multiple downstream computations sum at the parameter.}:

\begin{equation*}
    \displaystyle
    \Dp[\Ll]{\Wo} = \sum_{b\,\in\,\text{batch}}\left(\Dp[\Z]{\Wo}G\right)_b \in \mathbb{R}^{hd_v \times d_\text{model}}
\end{equation*}

We also have

\begin{align*}
    \Dp[\Z]{\aconcat} & = \begin{bmatrix}
        {\Wo_1}^\top & {\Wo_2}^\top & \cdots & {\Wo_h}^\top
    \end{bmatrix}\\
    & = \begin{bmatrix}
        \Dp[\Z]{\Aa_1} & \Dp[\Z]{\Aa_2} & \cdots & \Dp[\Z]{\Aa_h}
    \end{bmatrix}
\end{align*}

where we rewrote $\Wo$ as

\begin{align*}
    \Wo & = \begin{bmatrix}
        \Wo_1 & \Wo_2 & \cdots & \Wo_h
    \end{bmatrix}^\top, \quad \Wo_i \in \mathbb{R}^{d_v \times d_\text{model}}
\end{align*}

Now, since

\begin{align*}
    \Aa & = \begin{bmatrix}
        [\Aa_1] & [\Aa_2] & \cdots & [\Aa_h]
    \end{bmatrix} \in \mathbb{R}^{h\times N\times L_q \times d_v}\ (\Aa_i\in \mathbb{R}^{N\times L_q \times d_v})\\
\end{align*}

we have

\begin{align*}
    \Dp[\Z]{\Aa} & = \begin{bmatrix}
        [\Dp[\Z]{\Aa_1}] & [\Dp[\Z]{\Aa_2}] & \cdots & [\Dp[\Z]{\Aa_h}]
    \end{bmatrix}\\[2mm]
    & = \begin{bmatrix}
        [{\Wo_1}^\top] & [{\Wo_2}^\top] & \cdots & [{\Wo_h}^\top]
    \end{bmatrix}
\end{align*}

Finally, we can compute the gradient of the loss with respect to $\Aa$:

\begin{align*}
    \Dp[\Ll]{\Aa} & = G \Dp[\Z]{\Aa} \in \mathbb{R}^{N \times L_q \times h \times d_v}\\[-3mm]
\end{align*}

which we compute in NumPy as follows

\begin{minted}{python}

grad["dZ_da'"] = np.moveaxis(self.Wo, -1, -2) # (d_model, d_v * h)
grad["dZ_da"]  = np.stack(
                    np.split(grad["dZ_da'"], num_heads, axis=-1)
                 ) # (h, d_model, d_v)
grad["dL_da"] = np.dot(upstream_grad, grad["dZ_da"]) # (N, L_q, h, d_v)
                                             #   = (N, L_q, d_model) . (h, d_model, d_v)
grad["dL_da"] = np.moveaxis(grad["dL_da"], -2, 0) # (h, N, L_q, d_v)

\end{minted}

In the last equation, $G = \Dp[\Ll]{\Z}$ is the upstream gradient.

\subsection{Scaled dot product attention}


\subsubsection{Attention-weighted values}

\newcommand{\aforward}[0]{
\Aa & = \A\V \in \mathbb{R}^{h\times N \times L_q \times d_v}
}
\newcommand{\dadA}[0]{
\Dp[\Aa]{\A} & = \V^\top \in \mathbb{R}^{h\times N \times d_v \times L_k}
}
\newcommand{\dadV}[0]{
\Dp[\Aa]{\V} = \A^\top \in \mathbb{R}^{h\times N \times L_k \times L_q}
}
\newcommand{\dLdA}[0]{
\Dp[\Ll]{\A} & = G\Dp[\Aa]{\A} \in \mathbb{R}^{h\times N \times L_q \times L_k}
}
\newcommand{\dLdV}[0]{
\Dp[\Ll]{\V} = \Dp[\Aa]{\V}G \in \mathbb{R}^{h\times N \times L_k \times d_v}
}


\begin{align*}
    \aforward &&\tag*{(forward pass)} \\[2mm]
    \dadA, \quad \dadV && \tag*{(local gradients)}\\[2mm]
    \dLdA, \quad \dLdV && \tag*{(upstream of lower operations)}\\
\end{align*}

where $G = \Dp[\Ll]{\Aa}$ is the upstream gradient.


\subsubsection{Masked softmax}

Please see the PDF on \href{../utils/softmax_gradient.pdf}{the softmax function and its gradient}. As a reminder, here's the gradient with respect to the masked softmax inputs of the loss:

\begin{align*}
    \A = \text{softmax}(\Vector{S}) && \tag*{(forward pass)}\\
    \Dp[\Ll]{\Aa} = (G - G\cdot\Vector{p})\odot\Vector{p}\in\mathbb{R}^{h\times N\times L_q \times d_v} && \tag*{(upstream of lower operations)}\\
\end{align*}

\subsubsection{Scaled dot product}

\newcommand{\Sforward}[0]{
s = \frac{1}{\sqrt{d_K}}, \quad \Vector{S} & = s\Q\K^\top \in \mathbb{R}^{h\times N\times L_q\times L_k}
}
\newcommand{\dSdQ}[0]{
\Dp[\Vector{S}]{\Q} & = s\K \in \mathbb{R}^{h\times N \times L_k \times d_k}
}
\newcommand{\dSdK}[0]{
\Dp[\Vector{S}]{\K} = s\Q \in \mathbb{R}^{h\times N \times L_q \times d_k}
}
\newcommand{\dLdQ}[0]{
\Dp[\Ll]{\Q} & = sG\K \in \mathbb{R}^{h\times N \times L_q \times d_k}
}
\newcommand{\dLdK}[0]{
\Dp[\Ll]{\K} = sG^\top\Q \in \mathbb{R}^{h\times N \times L_k \times d_k}
}
\begin{align*}
    \Sforward &&\tag*{(forward pass)} \\[2mm]
    \dSdQ, \quad \dSdK && \tag*{(local gradients)}\\[2mm]
    \dLdQ, \quad \dLdK && \tag*{(upstream of lower operations)}\\
\end{align*}

\subsection{In projections}

For each head $i$, we have:

\newcommand{\Qforward}[0]{
\Q_i = \Vector{q}\Wqi, \quad \K_i & = \Vector{k}\Wki, \quad \V_i = \Vector{v}\Wvi
}
\newcommand{\dQdq}[0]{
\Dp[\Q_i]{\Vector{q}} = {\Wqi}^\top
}
\newcommand{\dKdk}[0]{
\Dp[\K_i]{\Vector{k}} = {\Wki}^\top
}
\newcommand{\dVdv}[0]{
\Dp[\V_i]{\Vector{v}} = {\Wvi}^\top
}
\newcommand{\dQdWq}[0]{
\Dp[\Q_i]{\Wqi} = \Vector{q}^\top
}
\newcommand{\dKdWk}[0]{
\Dp[\K_i]{\Wki} = \Vector{k}^\top
}
\newcommand{\dVdWv}[0]{
\Dp[\V_i]{\Wvi} = \Vector{v}^\top
}
\newcommand{\dLdq}[0]{
\Dp[\Ll]{\Vector{q}} = G\Dp[\Q_i]{\Vector{q}}
}
\newcommand{\dLdk}[0]{
\Dp[\Ll]{\Vector{k}} = G\Dp[\Q_i]{\Vector{q}}
}
\newcommand{\dLdv}[0]{
\Dp[\Ll]{\Vector{v}} = G\Dp[\Q_i]{\Vector{q}}
}
\newcommand{\dLdWqi}[0]{
\Dp[\Ll]{\Wqi} = \Dp[\Q_i]{\Wqi}G
}
\newcommand{\dLdWki}[0]{
\Dp[\Ll]{\Wki} = \Dp[\K_i]{\Wki}G
}
\newcommand{\dLdWvi}[0]{
\Dp[\Ll]{\Wvi} = \Dp[\V_i]{\Wvi}G
}
\begin{align*}
    \Qforward &&\tag*{(forward pass)} \\[7mm]
    \dQdq, \quad & \dKdk, \quad \dVdv && \tag*{(local gradients)}\\[2mm]
    \dQdWq, \quad & \dKdWk, \quad \dVdWv && \tag*{(local gradients)}\\[5mm]
    \dLdq, \quad & \dLdk, \quad \dLdv && \tag*{(upstream of lower operations)}\\[3mm]
    \dLdWqi, \quad & \dLdWki, \quad \dLdWvi && \tag*{(upstream of lower operations)}\\
\end{align*}

Now, to pack all of this into vectorized NumPy operations, we rewrite everything without the subscripts while being very careful about the shapes. We derive this for $\Vector{q}$ and $\Wq$ only as the equations for the rest are similar.

\newcommand{\Qprimeforward}{
\Q' & = \Vector{q}\Wq \in \mathbb{R}^{N \times L_q \times h \times d_k}
}
\renewcommand{\Qforward}{
\Q & = \texttt{np.moveaxis}(\Q', -2, 0) \in \mathbb{R}^{h \times N \times L_q \times d_k}\ (\ni G)
}
\renewcommand{\dQdq}{
\Dp[\Q]{\Vector{q}} & = \Wq^\top \in \mathbb{R}^{h\times d_k\times d_\text{model}}
}
\renewcommand{\dQdWq}{
\Dp[\Q]{\Wq} = \Vector{q}^\top \in \mathbb{R}^{N\times d_\text{model} \times L_q}
}
\begin{align*}
    \Qprimeforward && \tag*{(forward pass)} \\[3mm]
    \Qforward && \tag*{(forward pass)} \\[3mm]
    \dQdq,\quad \dQdWq && \tag*{(local gradients)}\\
\end{align*}

Here comes the tricky part: in the per-head equation we saw that $\dLdq$, but this will not give the correct shape if we simply carry out the product as follows:

\begin{minted}{python}
grad["dL_dq"] = G @ grad["dQ_dq"] # (h, N, L_q, d_k) @ (h, d_k, d_model)
                                  #  won't run; shapes don't match!
\end{minted}

the \texttt{@} operator won't complete properly because after it lines up the tensors on the last two axes ($L_q \times d_k$ and $d_k\times d_\text{model}$), the remaining axes are neither identical nor broadcastable (expected both to be either $h \times N$ or $h$ or $N$, but got $h\times N$ vs. $h$). Instead, we should be adding an axis to allow NumPy's broadcasting rules to kick in:


\renewcommand{\dQdq}{
\Dp[\Q]{\Vector{q}} & = \Wq^\top \in \mathbb{R}^{h\times 1 \times d_k\times d_\text{model}}
}
\renewcommand{\dQdWq}{
\Dp[\Q]{\Wq} = \Vector{q}^\top \in \mathbb{R}^{1\times N\times d_\text{model} \times L_q}
}
\renewcommand{\dLdq}{
    \Dp[\Ll]{\Vector{q}} & = G\Dp[\Q]{\Vector{q}} \in \mathbb{R}^{h\times N\times L_q \times d_\text{model}}
}
\newcommand{\dLdWq}{
    \Dp[\Ll]{\Wq} = \Dp[\Q]{\Wq}G \in \mathbb{R}^{h\times N\times d_\text{model}\times d_k}
}
\begin{align*}
    \dQdq, \quad \dQdWq && \tag*{(local gradients)} \\[5mm]
    \dLdq, \quad \dLdWq && \tag*{(upstream of lower operations)}
\end{align*}

This makes it so that the matrices represented by the final two dimensions interact independently for each head and batch.\\

Finally to make sure the dimensions are correct, we should be summing along the batch axis:

\renewcommand{\dLdq}{
    \Dp[\Ll]{\Vector{q}} & = \sum_{b\,\in\,\text{batch}}\left(G\Dp[\Q]{\Vector{q}}\right)_b \in \mathbb{R}^{h\times L_q \times d_\text{model}}
}
\renewcommand{\dLdWq}{
    \Dp[\Ll]{\Wq} = \sum_{b\,\in\,\text{batch}}\left(\Dp[\Q]{\Wq}G\right)_b \in \mathbb{R}^{h\times d_\text{model}\times d_k}
}
\begin{align*}
    \dLdq, \quad \dLdWq\\
\end{align*}

Similar arguments hold for the other matrices. -->

$$
\newcommand{\smx}[1]{\mathrm{softmax}(#1)}
\newcommand{\Vector}[1]{\mathbf{#1}}
\newcommand{\D}[1]{\frac{d}{d#1}}
\newcommand{\Dp}[2][]{\frac{\partial#1}{\partial#2}}
\newcommand{\smxv}[1]{\mathrm{softmax}(\Vector{#1})}
\newcommand{\W}[1]{\Vector{W^{#1}}}
\newcommand{\Wq}[0]{\Vector{W^q}}
\newcommand{\Wqi}[0]{\Vector{W}_i^\Vector{q}}
\newcommand{\Wk}[0]{\Vector{W^k}}
\newcommand{\Wki}[0]{\Vector{W}_i^\Vector{k}}
\newcommand{\Wv}[0]{\Vector{W^v}}
\newcommand{\Wvi}[0]{\Vector{W}_i^\Vector{v}}
\newcommand{\Wo}[0]{\Vector{W^o}}
\newcommand{\Aa}[0]{\Vector{a}}
\newcommand{\Z}[0]{\Vector{Z}}
\newcommand{\Q}[0]{\Vector{Q}}
\newcommand{\K}[0]{\Vector{K}}
\newcommand{\V}[0]{\Vector{V}}
\newcommand{\A}[0]{\Vector{A}}
\newcommand{\Ll}[0]{\mathcal{L}}
$$

# Full Derivation of Multihead Attention

Below, I'll be repeating the same idea multiple times, progressively adding bells and whistles to hopefully make this easier to follow. The calculations are presented in tensor (N-D array) format to make them directly compatible with NumPy, with $N$ referring to the number of batches.

## Attention

The core of the Transformer architecture is the attention function, which is surprisingly simple and straightforward to describe. Attention is a function of three inputs arguments (which are matrices) and one output argument (which is a matrix), and can be separated into three semantically distinct steps:

1. In-projections
2. Scaled dot-product attention
3. Out-projection

Note that these three steps are just names we're using to partition the operations of the attention function in an easier to digest way. Attention is just the composition of several function. These the names refer to the following equations:

1. In-projections

    $$
    \Vector{Q} = \Vector{q}\Vector{W^q}, \quad \Vector{K} = \Vector{k}\Vector{W^k}, \quad \Vector{V} = \Vector{v}\Vector{W^v}
    $$

    Here, the $\Vector{W}$ superscripts little $^\Vector{q}$, little $^\Vector{k}$, and little $^\Vector{v}$ are not indexes---they are just part of the variable's name.

2. Scaled dot-product attention

    $$
    \Vector{A} = \smx{\frac{\Vector{Q}\Vector{K}^\top}{\sqrt{d_k}}}\Vector{V}
    $$

    where $d_k$ is a scalar hyperparameter we will get to later.

3. Out-projection

    $$
    \Vector{Z} = \Vector{A}\Vector{W^o}
    $$

    Here, too, the superscript little $^\Vector{o}$ is part of the variable's name and is not an index.

## Multiheaded Attention

The transformer uses multiheaded attention which is just a fancy-sounding way to say it repeats the above operations multiple times using different values for the four $\Vector{W}$ parameter matrices.

The transformer multiheaded attention boils down to the following equations, where we now use the $i=1,\ldots, h$ subscript *indexes* to distinguish between the different *heads* and write the whole thing a bit more formally:

$$
\begin{align}
    \Vector{Q}_i = \Vector{q}\Vector{W}_i^\Vector{^q}, \quad \Vector{K}_i = \Vector{k}\Vector{W}_i^\Vector{^k}, \quad \Vector{V}_i = \Vector{v}\Vector{W}_i^\Vector{^v}\\
    \Vector{A}_i = \smx{\frac{\Vector{Q}_i\Vector{K}_i^\top}{\sqrt{d_k}}}\Vector{V}_i\\
    \Vector{Z} = \begin{bmatrix}
        \Vector{A}_1 & \Vector{A}_2 & \cdots & \Vector{A}_h
    \end{bmatrix}\Vector{W^o}
\end{align}
$$

Putting it all together we get the following (multiheaded attention in a single equation):

$$
\begin{align}
    \Vector{Z}(\Vector{q}, \Vector{k}, \Vector{v}) = \begin{bmatrix}
        \displaystyle\smx{\frac{\Vector{q}\Vector{W}_1^\Vector{^q}(\Vector{k}\Vector{W}_1^\Vector{^k})^\top}{\sqrt{d_k}}}\Vector{v}\Vector{W}_1^\Vector{^v}\\[6mm]
        \displaystyle\smx{\frac{\Vector{q}\Vector{W}_2^\Vector{^q}(\Vector{k}\Vector{W}_2^\Vector{^k})^\top}{\sqrt{d_k}}}\Vector{v}\Vector{W}_2^\Vector{^v}\\[6mm]
        \vdots\\[6mm]
        \displaystyle\smx{\frac{\Vector{q}\Vector{W}_h^\Vector{^q}(\Vector{k}\Vector{W}_h^\Vector{^k})^\top}{\sqrt{d_k}}}\Vector{v}\Vector{W}_h^\Vector{^v}\\[6mm]
    \end{bmatrix}^\top\Vector{W^o} \label{eqn:multihead_attn}
\end{align}
$$
<aside>
<p>
This is it! One equation to rule them all!
</p>
</aside>

where the three-input $\Vector{Z}$ (the multiheaded attention with $h$ heads) is parameterized by $3h+1$ matrices: $\Vector{W}_i\Vector{^q}$, $\Vector{W}_i\Vector{^k}$, $\Vector{W}_i\Vector{^v}$, and $\Vector{W^o}$, $i=1,\ldots,h$.

The "hardest" thing about implementing multiheaded attention from scratch is managing the matrix shapes effectively so as to make the matrix multiplications efficient.

## Full description of multiheaded attention operations

### In projections

| Inputs | Parameters | Forward pass operations | Outputs |
|--------|------------|------------------------|---------|
| $\Vector{q} \in \mathbb{R}^{N\times L_q\times d_\text{model}}$ | $\Vector{W_q} \in \mathbb{R}^{h\times d_\text{model}\times d_k}$ | `Q = np.dot(q,Wq)` <br> `Q = np.moveaxis(Q, -2, 0)` | $\Vector{Q} \in \mathbb{R}^{h\times N\times L_q\times d_k}$ |
| $\Vector{k} \in \mathbb{R}^{N\times L_k\times kdim}$ | $\Vector{W_k} \in \mathbb{R}^{h\times kdim\times d_k}$ | `K = np.dot(k,Wk)` <br> `K = np.moveaxis(K, -2, 0)` | $\Vector{K} \in \mathbb{R}^{h\times N\times L_k\times d_k}$ |
| $\Vector{v} \in \mathbb{R}^{N\times L_k\times vdim}$ | $\Vector{W_v} \in \mathbb{R}^{h\times vdim\times d_v}$ | `V = np.dot(v,Wv)` <br> `V = np.moveaxis(V, -2, 0)` | $\Vector{V} \in \mathbb{R}^{h\times N\times L_k\times d_v}$ |

Here, $d_\text{model} = hd_k$ is the *dimensionality of the model*, i.e., the dimensionality of the token embeddings coming in. We typically choose $d_\text{model}$ and $h$ and calculate $d_k$ as $\frac{d_\text{model}}{h}$, implying we require $d_\text{model}$ to be divisible by $h$.

### Scaled dot product attention

#### Scaled dot product

| Inputs | Forward pass operations | Outputs |
|--------|------------------------|---------|
| $\Vector{Q} \in \mathbb{R}^{h\times N\times L_q\times d_k}$ <br> $\Vector{K} \in \mathbb{R}^{h\times N\times L_k\times d_k}$ | `s = 1 / np.sqrt(d_k)` <br> `S = Q @ np.swapaxes(K, -1, -2) * s` | $\Vector{S}\in\mathbb{R}^{h\times N\times L_q \times L_k}$ |

#### Masked Softmax

For the softmax gradient, see [softmax and its gradient post]({% post_url 2026-03-05-softmax-gradient %}).

| Inputs | Forward pass operations | Outputs |
|--------|------------------------|---------|
| $\Vector{S}\in\mathbb{R}^{h\times N\times L_q \times L_k}$ | See softmax and its gradient PDF | $\Vector{A}\in\mathbb{R}^{h\times N\times L_q \times L_k}$ |

#### Attention-weighted values

$$\newcommand{\aconcat}[0]{\Aa'}$$

| Inputs | Forward pass operations | Outputs |
|--------|------------------------|---------|
| $\A\in\mathbb{R}^{h\times N\times L_q \times L_k}$ <br> $\Vector{V} \in \mathbb{R}^{h\times N\times L_k\times d_v}$ | `a = A @ V # shape (h, N, L_q, d_v)` <br> `a' = np.concatenate(a, axis=-1)` | $\Aa'\in\mathbb{R}^{N\times L_q \times hd_v}$ |

We used a different name $\aconcat$ for the concatenated form of $\Aa$ because later this will make reasoning about the gradients clearer.

### Out projection

| Inputs | Parameters | Forward pass operations | Outputs |
|--------|------------|------------------------|---------|
| $\Vector{a}'\in\mathbb{R}^{N\times L_q \times hd_v}$ | $\Vector{W_o} \in \mathbb{R}^{hd_v\times d_\text{model}}$ | `Z = a' @ Wo` | $\Vector{Z} \in \mathbb{R}^{N\times L_q\times d_\text{model}}$ |

Note that the output $\Vector{Z}$ has the exact same dimensionality as the input $\Vector{q}$, and this is by design, which is why we can add the residual stream and stack multiple Multiheaded Attention blocks sequentially. To see this graphically for a single head, see Figure 1.

*Figure 1: **Attention function and the residual stream**. This figure visually describes the function discussed in this document. It also shows the residual stream (bottom arrow and the summation $\oplus$) and the (optional) dropout layer. Note that this representation obfuscates some intermediate steps which have been made explicit in the equations of this document.*

## The gradients (backward pass)

We begin deriving the gradients from the last operation, the out projection, and work our way back to the inputs. For any operation $y = f(x)$, we will use the shorthand

$$
G = \Dp[\mathcal{L}]{y} \in \mathcal{X}
$$

for the gradient of the final loss function $\mathcal{L}$ with respect to the current operation's output $y$. $\mathcal{X}$ is the set to which both $G$ and $y$ belong (determines dimensionality of both). We refer to $G$ as the "upstream gradient."

### Out projection

$$
\newcommand{\Zforward}{
\Z & = \aconcat\Wo = \begin{bmatrix}
        \Aa_1 & \Aa_2 & \cdots & \Aa_h
    \end{bmatrix} \Wo \in \mathbb{R}^{N\times L_q\times d_\text{model}}
}
\newcommand{\dZda}{
\Dp[\Z]{\aconcat} = \Wo^\top \in \mathbb{R}^{d_\text{model} \times hd_v}
}
\newcommand{\dZdWo}{
\Dp[\Z]{\Wo} = \aconcat^\top \in \mathbb{R}^{N\times hd_v \times L_q}
}
\newcommand{\dLda}{
\Dp[\Ll]{\aconcat} = G\Dp[\Z]{\aconcat} \in\mathbb{R}^{N\times L_q \times hd_v}
}
\newcommand{\dLdWo}{
\Dp[\Ll]{\Wo} = \Dp[\Z]{\Wo}G \in \mathbb{R}^{N\times hd_v \times d_\text{model}}
}
$$

$$
\begin{align*}
    \Zforward \\[3mm]
    \dZda & ,\quad \dZdWo \\[3mm]
    \dLda & ,\quad \dLdWo \\
\end{align*}
$$

To match the dimensionality of $\Dp[\Ll]{\Wo}$ with that of $\Wo$, we must sum over the batch dimension:
<aside>
<p>
This is not the *real* reason but it sounds like a better mnemonic. The real reason is that by performing the preceding calculations, we have computed $N$ different gradients as a result of the $N$ batches, and that, by chain rule, gradients flowing in to a parameter from multiple downstream computations sum at the parameter.
</p>
</aside>

$$
\Dp[\Ll]{\Wo} = \sum_{b\,\in\,\text{batch}}\left(\Dp[\Z]{\Wo}G\right)_b \in \mathbb{R}^{hd_v \times d_\text{model}}
$$

We also have

$$
\begin{align*}
    \Dp[\Z]{\aconcat} & = \begin{bmatrix}
        {\Wo_1}^\top & {\Wo_2}^\top & \cdots & {\Wo_h}^\top
    \end{bmatrix}\\
    & = \begin{bmatrix}
        \Dp[\Z]{\Aa_1} & \Dp[\Z]{\Aa_2} & \cdots & \Dp[\Z]{\Aa_h}
    \end{bmatrix}
\end{align*}
$$

where we rewrote $\Wo$ as

$$
\begin{align*}
    \Wo & = \begin{bmatrix}
        \Wo_1 & \Wo_2 & \cdots & \Wo_h
    \end{bmatrix}^\top, \quad \Wo_i \in \mathbb{R}^{d_v \times d_\text{model}}
\end{align*}
$$

Now, since

$$
\begin{align*}
    \Aa & = \begin{bmatrix}
        [\Aa_1] & [\Aa_2] & \cdots & [\Aa_h]
    \end{bmatrix} \in \mathbb{R}^{h\times N\times L_q \times d_v}\ (\Aa_i\in \mathbb{R}^{N\times L_q \times d_v})\\
\end{align*}
$$

we have

$$
\begin{align*}
    \Dp[\Z]{\Aa} & = \begin{bmatrix}
        [\Dp[\Z]{\Aa_1}] & [\Dp[\Z]{\Aa_2}] & \cdots & [\Dp[\Z]{\Aa_h}]
    \end{bmatrix}\\[2mm]
    & = \begin{bmatrix}
        [{\Wo_1}^\top] & [{\Wo_2}^\top] & \cdots & [{\Wo_h}^\top]
    \end{bmatrix}
\end{align*}
$$

Finally, we can compute the gradient of the loss with respect to $\Aa$:

$$
\begin{align*}
    \Dp[\Ll]{\Aa} & = G \Dp[\Z]{\Aa} \in \mathbb{R}^{N \times L_q \times h \times d_v}\\[-3mm]
\end{align*}
$$

which we compute in NumPy as follows

```python
grad["dZ_da'"] = np.moveaxis(self.Wo, -1, -2) # (d_model, d_v * h)
grad["dZ_da"]  = np.stack(
                    np.split(grad["dZ_da'"], num_heads, axis=-1)
                 ) # (h, d_model, d_v)
grad["dL_da"] = np.dot(upstream_grad, grad["dZ_da"]) # (N, L_q, h, d_v)
                                             #   = (N, L_q, d_model) . (h, d_model, d_v)
grad["dL_da"] = np.moveaxis(grad["dL_da"], -2, 0) # (h, N, L_q, d_v)
```

In the last equation, $G = \Dp[\Ll]{\Z}$ is the upstream gradient.

### Scaled dot product attention

#### Attention-weighted values

$$
\newcommand{\aforward}{
\Aa & = \A\V \in \mathbb{R}^{h\times N \times L_q \times d_v}
}
\newcommand{\dadA}{
\Dp[\Aa]{\A} & = \V^\top \in \mathbb{R}^{h\times N \times d_v \times L_k}
}
\newcommand{\dadV}{
\Dp[\Aa]{\V} = \A^\top \in \mathbb{R}^{h\times N \times L_k \times L_q}
}
\newcommand{\dLdA}{
\Dp[\Ll]{\A} & = G\Dp[\Aa]{\A} \in \mathbb{R}^{h\times N \times L_q \times L_k}
}
\newcommand{\dLdV}{
\Dp[\Ll]{\V} = \Dp[\Aa]{\V}G \in \mathbb{R}^{h\times N \times L_k \times d_v}
}
$$

$$
\begin{align*}
    \aforward &&\tag*{(forward pass)} \\[2mm]
    \dadA, \quad \dadV && \tag*{(local gradients)}\\[2mm]
    \dLdA, \quad \dLdV && \tag*{(upstream of lower operations)}\\
\end{align*}
$$

where $G = \Dp[\Ll]{\Aa}$ is the upstream gradient.

#### Masked softmax

Please see the PDF on [the softmax function and its gradient](../utils/softmax_gradient.pdf). As a reminder, here's the gradient with respect to the masked softmax inputs of the loss:

$$
\begin{align*}
    \A = \text{softmax}(\Vector{S}) && \tag*{(forward pass)}\\
    \Dp[\Ll]{\Aa} = (G - G\cdot\Vector{p})\odot\Vector{p}\in\mathbb{R}^{h\times N\times L_q \times d_v} && \tag*{(upstream of lower operations)}\\
\end{align*}
$$

#### Scaled dot product

$$
\newcommand{\Sforward}{
s = \frac{1}{\sqrt{d_K}}, \quad \Vector{S} & = s\Q\K^\top \in \mathbb{R}^{h\times N\times L_q\times L_k}
}
\newcommand{\dSdQ}{
\Dp[\Vector{S}]{\Q} & = s\K \in \mathbb{R}^{h\times N \times L_k \times d_k}
}
\newcommand{\dSdK}{
\Dp[\Vector{S}]{\K} = s\Q \in \mathbb{R}^{h\times N \times L_q \times d_k}
}
\newcommand{\dLdQ}{
\Dp[\Ll]{\Q} & = sG\K \in \mathbb{R}^{h\times N \times L_q \times d_k}
}
\newcommand{\dLdK}{
\Dp[\Ll]{\K} = sG^\top\Q \in \mathbb{R}^{h\times N \times L_k \times d_k}
}
$$

$$
\begin{align*}
    \Sforward &&\tag*{(forward pass)} \\[2mm]
    \dSdQ, \quad \dSdK && \tag*{(local gradients)}\\[2mm]
    \dLdQ, \quad \dLdK && \tag*{(upstream of lower operations)}\\
\end{align*}
$$

### In projections

For each head $i$, we have:

$$
\newcommand{\Qforward}{
\Q_i = \Vector{q}\Wqi, \quad \K_i & = \Vector{k}\Wki, \quad \V_i = \Vector{v}\Wvi
}
\newcommand{\dQdq}{
\Dp[\Q_i]{\Vector{q}} = {\Wqi}^\top
}
\newcommand{\dKdk}{
\Dp[\K_i]{\Vector{k}} = {\Wki}^\top
}
\newcommand{\dVdv}{
\Dp[\V_i]{\Vector{v}} = {\Wvi}^\top
}
\newcommand{\dQdWq}{
\Dp[\Q_i]{\Wqi} = \Vector{q}^\top
}
\newcommand{\dKdWk}{
\Dp[\K_i]{\Wki} = \Vector{k}^\top
}
\newcommand{\dVdWv}{
\Dp[\V_i]{\Wvi} = \Vector{v}^\top
}
\newcommand{\dLdq}{
\Dp[\Ll]{\Vector{q}} = G\Dp[\Q_i]{\Vector{q}}
}
\newcommand{\dLdk}{
\Dp[\Ll]{\Vector{k}} = G\Dp[\Q_i]{\Vector{q}}
}
\newcommand{\dLdv}{
\Dp[\Ll]{\Vector{v}} = G\Dp[\Q_i]{\Vector{q}}
}
\newcommand{\dLdWqi}{
\Dp[\Ll]{\Wqi} = \Dp[\Q_i]{\Wqi}G
}
\newcommand{\dLdWki}{
\Dp[\Ll]{\Wki} = \Dp[\K_i]{\Wki}G
}
\newcommand{\dLdWvi}{
\Dp[\Ll]{\Wvi} = \Dp[\V_i]{\Wvi}G
}
$$

$$
\begin{align*}
    \Qforward &&\tag*{(forward pass)} \\[7mm]
    \dQdq, \quad & \dKdk, \quad \dVdv && \tag*{(local gradients)}\\[2mm]
    \dQdWq, \quad & \dKdWk, \quad \dVdWv && \tag*{(local gradients)}\\[5mm]
    \dLdq, \quad & \dLdk, \quad \dLdv && \tag*{(upstream of lower operations)}\\[3mm]
    \dLdWqi, \quad & \dLdWki, \quad \dLdWvi && \tag*{(upstream of lower operations)}\\
\end{align*}
$$

Now, to pack all of this into vectorized NumPy operations, we rewrite everything without the subscripts while being very careful about the shapes. We derive this for $\Vector{q}$ and $\Wq$ only as the equations for the rest are similar.

$$
\newcommand{\Qprimeforward}{
\Q' & = \Vector{q}\Wq \in \mathbb{R}^{N \times L_q \times h \times d_k}
}
\renewcommand{\Qforward}{
\Q & = \texttt{np.moveaxis}(\Q', -2, 0) \in \mathbb{R}^{h \times N \times L_q \times d_k}\ (\ni G)
}
\renewcommand{\dQdq}{
\Dp[\Q]{\Vector{q}} & = \Wq^\top \in \mathbb{R}^{h\times d_k\times d_\text{model}}
}
\renewcommand{\dQdWq}{
\Dp[\Q]{\Wq} = \Vector{q}^\top \in \mathbb{R}^{N\times d_\text{model} \times L_q}
}
$$

$$
\begin{align*}
    \Qprimeforward && \tag*{(forward pass)} \\[3mm]
    \Qforward && \tag*{(forward pass)} \\[3mm]
    \dQdq,\quad \dQdWq && \tag*{(local gradients)}\\
\end{align*}
$$

Here comes the tricky part: in the per-head equation we saw that $\dLdq$, but this will not give the correct shape if we simply carry out the product as follows:

```python
grad["dL_dq"] = G @ grad["dQ_dq"] # (h, N, L_q, d_k) @ (h, d_k, d_model)
                                  #  won't run; shapes don't match!
```

the `@` operator won't complete properly because after it lines up the tensors on the last two axes ($L_q \times d_k$ and $d_k\times d_\text{model}$), the remaining axes are neither identical nor broadcastable (expected both to be either $h \times N$ or $h$ or $N$, but got $h\times N$ vs. $h$). Instead, we should be adding an axis to allow NumPy's broadcasting rules to kick in:

$$
\renewcommand{\dQdq}{
\Dp[\Q]{\Vector{q}} & = \Wq^\top \in \mathbb{R}^{h\times 1 \times d_k\times d_\text{model}}
}
\renewcommand{\dQdWq}{
\Dp[\Q]{\Wq} = \Vector{q}^\top \in \mathbb{R}^{1\times N\times d_\text{model} \times L_q}
}
\renewcommand{\dLdq}{
    \Dp[\Ll]{\Vector{q}} & = G\Dp[\Q]{\Vector{q}} \in \mathbb{R}^{h\times N\times L_q \times d_\text{model}}
}
\newcommand{\dLdWq}{
    \Dp[\Ll]{\Wq} = \Dp[\Q]{\Wq}G \in \mathbb{R}^{h\times N\times d_\text{model}\times d_k}
}
$$

$$
\begin{align*}
    \dQdq, \quad \dQdWq && \tag*{(local gradients)} \\[5mm]
    \dLdq, \quad \dLdWq && \tag*{(upstream of lower operations)}
\end{align*}
$$

This makes it so that the matrices represented by the final two dimensions interact independently for each head and batch.

Finally to make sure the dimensions are correct, we should be summing along the batch axis:

$$
\renewcommand{\dLdq}{
    \Dp[\Ll]{\Vector{q}} & = \sum_{b\,\in\,\text{batch}}\left(G\Dp[\Q]{\Vector{q}}\right)_b \in \mathbb{R}^{h\times L_q \times d_\text{model}}
}
\renewcommand{\dLdWq}{
    \Dp[\Ll]{\Wq} = \sum_{b\,\in\,\text{batch}}\left(\Dp[\Q]{\Wq}G\right)_b \in \mathbb{R}^{h\times d_\text{model}\times d_k}
}
$$

$$
\begin{align*}
    \dLdq, \quad \dLdWq\\
\end{align*}
$$

Similar arguments hold for the other matrices.
