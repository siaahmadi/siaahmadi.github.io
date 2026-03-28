---
layout: distill
title: Can you get home twice as fast in two cars instead of one? Scaling up autoregressive inference in LLMs
description: We will explore the Tiered Transformer, a novel architecture that enables scaling up inference in large language models without sacrificing the autoregressive factorization of the output's joint probability distribution.
tags:
  - ai
  - optimization
  - neural-network
  - llm
  - systems
  - transformer
categories: research, AI
giscus_comments: false
date: 2026-03-14
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
    url: "https://github.com/siaahmadi/tiered_transformer"
    affiliations:
      name: Independent Researcher

bibliography: 2026-03-14-scaling-llm-inference.bib

# Optionally, you can add a table of contents to your post.
# NOTES:
#   - make sure that TOC names match the actual section names
#     for hyperlinks within the post to work correctly.
#   - we may want to automate TOC generation in the future using
#     jekyll-toc plugin (https://github.com/toshimaru/jekyll-toc).
toc:
  - name: A fun little experiment
  - name: Intro
  - name: A solution
  - name: Here's an example
  - name: From Outline to Architecture
  - name: Isn't this just an encoder-decoder transformer?
  - name: Throughput gains
  - name: Conclusion


# Below is an example of injecting additional post-specific styles.
# If you use this post as a template, delete this _styles block.
# _styles: >
#   .fake-img {
#     background: #bbb;
#     border: 1px solid rgba(0, 0, 0, 0.1);
#     box-shadow: 0 0px 4px rgba(0, 0, 0, 0.1);
#     margin-bottom: 12px;
#   }
#   .fake-img p {
#     font-family: monospace;
#     color: white;
#     text-align: left;
#     margin: 12px 0;
#     text-align: center;
#   }
#     font-size: 16px;
---

<!-- # THIS POST IS CURRENTLY IN DEVELOPMENT. PLEASE COME BACK SOON! -->

To answer the question in the post title---and in case there was any doubt 😅---the answer is no. Even if you were to attempt the ill-advised trick of splitting yourself in half to be placed in two separate cars, you still won't get home twice as fast, unfortunately. Crucially, this does not even depend on splitting yourself in half: if you and your partner were going home from the opera in two separate cars, you won't get home any faster than if you went in the same car.

Obviously, this is a rather facetious way to make a simple point: inherently sequential tasks cannot be completed faster by using additional resources. Large language model (LLM) response generation suffers from the same limitation. No matter how many GPUs you throw at the problem, generating a sequence of tokens one by one can take longer than you'd like.

In this post, I will focus on building a high-level intuition by introducing the architecture using analogies that complement the formal paper rather than just repeating its equations. If you're curious about implementation details, you will want to read the [companion preprint]({% link assets/pdf/Tiered_Transformer.pdf %}).

## A fun little experiment

Before we dive in, I want to share a fun little experiment. I used Google NotebookLM to generate a "podcast" from an earlier verison of the companion [preprint]( {% link assets/pdf/Tiered_Transformer.pdf %}). Take a listen:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/Tiered_Transformers_enable_parallel_text_generation.m4a" controls=true %}
    </div>
</div>
<div class="caption">
    An AI-generated "podcast" segment on the topic of this blog post.
</div>

## Intro

Think back to your last piece of writing longer than a single paragraph. Did you write it linearly from start to finish? Chances are, _not_.

Whether drafting a paper or a blog post, most of us probably don't sit down and produce a linear stream of consciousness from the first word to the last.

In practice, we start by sketching the skeleton, breaking the thesis into (semi-)independent yet interrelated sections that we can tackle in isolation. We might even do this recursively: sections might be split into paragraphs, and paragraphs into sentences, allowing us to build the work semi-independently as the ideas mature.

Among the many benefits of working in this way is that (at least in theory) you could deploy the task of writing each section to a different person. If you're writing a book, you can develop the main story, come up with the general outline of say 10 chapters, and then outsource the task of actually writing the chapters to 10 different ghost writers. In this way, you can actually finish your book 10 times as fast, even though you might write your 10 chapter outlines sequentially!

LLMs don't operate that way, however. Even when writing a very long response, LLMs produce output strictly linearly, in an autoregressive fashion: predicting one token at a time, conditioned only on the sequence that has come before it. We know that this approach produces impressively high quality, coherent output in many cases. However, it cannot be distributed among parallel workers. And as the capabilities of LLMs depend more and more on *compute-time scaling* as training data get more scarce, it becomes increasingly more important to make it possible to scale token generation.


## A solution

Now, what if we had a model that mimics how we write? The Tiered Transformer attempts to do just that. The Tiered Transformer architecture, explicitly separates the task of creating a high-level blueprint from that of generating tokens.

Like in the example of using ghostwriters to write individual chapters of a book, the Tiered transformer uses a dedicated module to generate the "chapter" outlines first. We call these individual high-level ideas "concepts."

Think of a concept as a high-dimensional vector that summarizes the core ideas that a sentence, paragraph, or chapter must convey. The "Concept Decoder" module in the Tiered Transformer architecture is tasked with autoregressively generating these concept embeddings. This is what we might do when we begin drafting an email, a paper, or a story.

Once the Concept Decoder is done generating the main ideas, the "Token Decoder" begins translating each high-dimensional concept embedding into tokens. Think of the Token Decoder as the ghost writer. Remember, the concept vectors were already generating in such a way that each is conditioned on the concepts that came before it (the concept decoder generated them autoregressively, just like regular LLMs operate). And that's why we can have multiple independent Token Decoder modules begin working on each concept in parallel. We will just have to wait until all Token Decoder's have finished generating their portion of the larger response.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/tiered_transformer_v1.gif" class="img-fluid z-depth-1" %}
    </div>
</div>
<div class="caption">
    The Tiered Transformer at work. The top row shows the Concept Decoder, and the bottom matrix shows the Token decoder. The gold circles represent the high-level concepts and the green ones the individual tokens. Scaling is possible because each concept's tokens depend on the previous tokens of *only that concept* and the previous high-level concepts (and *not their tokens*).
</div>



Now, notice that both steps of this procedure retain the autoregressive factorization: the Concept Decoder works autoregressively, and each Token Decoder also works autoregressively. This means that we can expect the Tiered Transformer to learn an exact distribution over the joint distribution of concepts and of tokens. This is an important property to preserve since previous work on parallelizing token generation in language models has shown that if tokens are generated using a conditional independence assumption, the output quality suffers.

If you squint a little, you might notice that this modular structure bears some resemblances to an autoencoder. In a traditional autoencoder, you map the input to some latent space and then unto this operation by mapping the latent space representations back to the original input space. The high-dimensional concept vector essentially acts as the informational bottleneck in the Tiered Transformer. The Token Decoder then steps into the role of the traditional decoder, unpacking this high-dimensional blueprint into a sequence of discrete tokens. The crucial difference, of course, is that we aren't compressing an existing sequence; our Concept Decoder is generating the bottleneck itself autoregressively from scratch, which is exactly how we preserve that all-important joint probability distribution.

### Here's an example

Let's take a look at an example. I want to look at the Introduction of one of my PhD papers, which is [publicly available here](https://doi.org/10.1038/s41467-024-54943-2). I asked [Grok](https://grok.com/) and [Claude](https://claude.ai) to summarize each paragraph, and here's what they wrote (I edited and merged them):

1. DG supports memory via sparse activity, pattern separation, and complex spatial working memory.
2. CA3 sharp-wave ripples depend on DG during working memory and enable sequence replay, and aid planning.
3. DG may correct sequence order in CA3; theta dominates active exploration and encoding.
4. Theta phase precession compresses behavioral sequences into theta cycle firing, enabling synaptic storage.
5. DG's contribution to CA3 phase precession remains computationally proposed but experimentally untested.
6. We found DG, not MEC, drives prospective coding in CA3 theta; new model explains oscillatory contributions.

Now, this somewhat captures what I had in mind when I was actually trying to write a paragraph. The process did involve additional details, such as planning and iterating on writing and re-writing the core ideas, but each individual iteration followed a similar routine: I kept asking myself _what am I trying to say?_ and _how should I string my words together to say that?_.

The above points can be compressed even further:

1. DG, broadly
2. DG vs. CA3 sharp-wave ripples
3. The broader circuit and role of theta
4. Theta sequences
5. The scientific gap
6. Our contributions

Now, it can be seen that each point (a) depends on the previous points, and (b) can be expanded into sentences in various ways.

I asked Claude this time to expand the first four of these into possible paragraph outlines:

1. **DG, broadly**: The dentate gyrus is a hippocampal subregion critical for forming distinct memory representations through sparse coding.
2. **DG vs. CA3 sharp-wave ripples**: DG inputs regulate CA3 sharp-wave ripples, high-frequency oscillatory events linked to memory consolidation and replay.
3. **The broader circuit and role of theta**: Beyond sharp-wave ripples, theta oscillations dominate hippocampal dynamics during active exploration and memory encoding.
4. **Theta sequences**: During theta, place cells fire in compressed sequential order reflecting the animal's past and future trajectory.


And finally, here's how it expanded them back to full paragraphs:

>The dentate gyrus is a hippocampal subregion critical for forming distinct memory representations through sparse coding. Situated at the entry point of the canonical hippocampal circuit, it receives the majority of its cortical drive via the perforant path and transforms dense, overlapping cortical inputs into sparse, highly differentiated activity patterns. This process, often referred to as pattern separation, is thought to minimize interference between similar memories by ensuring that distinct experiences are represented by non-overlapping neuronal ensembles. The unusually low baseline firing rates of dentate granule cells, combined with powerful feedback inhibition from local interneurons, are key cellular mechanisms that enforce this sparsity.
>
>DG inputs regulate CA3 sharp-wave ripples, high-frequency oscillatory events linked to memory consolidation and replay. CA3, the principal downstream target of the dentate gyrus via the mossy fiber pathway, possesses extensive recurrent collateral connections that endow it with autoassociative properties suited to pattern completion and sequence storage. Periods of behavioral quiescence and slow-wave sleep are punctuated by sharp-wave ripples, during which reactivation of previously encoded neuronal sequences is thought to transfer information to neocortical long-term stores. The degree to which the dentate gyrus gates or modulates this consolidation process through its projections to CA3 has become an important question for understanding how memory traces are stabilized over time.
>
>Beyond sharp-wave ripples, theta oscillations dominate hippocampal dynamics during active exploration and memory encoding. This 6–10 Hz rhythm, sustained by a network encompassing the medial septum and entorhinal cortex, imposes a recurring temporal structure on hippocampal activity that is thought to segregate encoding and retrieval processes across successive cycles. Both the dentate gyrus and CA3 receive prominent theta-modulated excitation from the medial entorhinal cortex, raising the question of how these two external inputs interact and whether they make separable contributions to CA3 network dynamics during behavior.
>
>During theta, place cells fire in compressed sequential order reflecting the animal's past and future trajectory. As the animal moves through an environment, hippocampal neurons with overlapping place fields discharge in a fixed order within each theta cycle, with each cell advancing toward earlier phases as the animal progresses through its field. The resulting spike sequences, compressed from behavioral timescales into windows compatible with synaptic plasticity, are candidate substrates for the rapid encoding of spatial trajectories. Determining which afferent inputs to CA3 are necessary for generating and maintaining this temporal organization is therefore essential for understanding how the hippocampal circuit constructs and stores sequential representations of experience.

This is obviously not as high-quality as the actual paper's introduction. For example, it does not even mention "phase precession" explicitly, and its vague description of it is not even accurate (e.g., "with each cell advancing toward earlier phases" -> the cells don't advance, it's their spike timing relative to theta oscillations that advances). And more importantly, it doesn't hit the main points that the paper is about to satisfactorily set up the scientific gap and our contributions.

Now, despite those caveats, this illustrates a main point: that the *core ideas* and *actual tokens* are somewhat separable (this is not necessarily my idea and there are serious people talking about this all the time, e.g., <d-cite key="fedorenko2024language"></d-cite>).

## From Outline to Architecture

So how do we generate the concepts? As a first-pass attempt, I used a pre-trained BERT model to extract feature representations. The Tiered Transformer is trained on BERT's [CLS] vectors. Because BERT is a bidirectional encoder, its [CLS] token effectively serves as a "summary" of the entire input sequence (or each sentence, or paragraph; that's a design choice). The Tiered Transformer is trained to predict these [CLS] vectors, effectively learning to generate the "outline" or "concept" of the text before parallelizing the token generation. The choice of BERT is not crucial here, which can be seen simply as a preprocessor. 

## Isn't this just an encoder-decoder transformer?

At first glance, this two-stage setup might look suspiciously like a standard encoder-decoder architecture, such as T5 or BART. In translation tasks, an encoder takes in the source text to generate the KV inputs to the cross-attention layer in the decoder. This KV input is what provides the right summary of the "context" (i.e., original text) which the decoder must utilize to generate the translation.

In our case, the Concept Decoder plays a similar role to that encoder, but with a critical distinction: it is strictly autoregressive with causal token masking. This allows the model to maintain the exact joint probability distribution of the sequence. Furthermore, instead of generating a single output to be used as the input to the K and V matrices of the decoder's cross-attention, it actually generates multiple embeddings that each depend on their respective predecessors.

In ongoing work, I am exploring an alternative design whereby the concept decoder generates the token decoder's KV inputs, rather than input embeddings. Stay tuned!

## Throughput gains

When all is said done, the Tiered Transformer was able reach a 72-fold token throughput compared to a single decoder-only architecture. It is worth noting that I observed this scaling with a relatively small proof-of-concept model (around 10 million parameters) so further testing is needed to assess the gains in more realistic settings.


<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/tiered_transformer_fig3_throughput.png" class="img-fluid z-depth-1" %}
    </div>
</div>
<div class="caption">
    The Tiered Transformer reaches a peak of 72x throughput gains over a sequential decoder-only transformer. The X-axis denotes the number of sequential forward passes, which scales linearly with the number of tokens generated in a standard autoregressive model.
</div>


I have more results in the preprint so feel free to take a look!

## Conclusion

Scaling autoregressive generation is fundamentally constrained by the sequential nature of one-token-at-a-time decoding. People typically outline high-level concepts before drafting the exact words. By taking inspiration from that, the Tiered Transformer breaks this bottleneck. Separating the generation of latent concepts from the decoding of discrete tokens allows us to parallelize inference without sacrificing learning the joint probability distribution.