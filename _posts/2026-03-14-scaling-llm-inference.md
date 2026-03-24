---
layout: distill
title: Can you get home twice as fast in two cars instead of one? Scaling up autoregressive inference in LLMs
description: We will explore a novel architecture that enables scaling up inference in large language models without sacrificing the autoregressive factorization of the output's joint probability distribution.
tags: ai, optimization, neural-network, llm, systems, transformer
categories: AI, research
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
  - name: The model
  - name: Inference
    subsections:
      - name: Training
  - name: Limitations
  - name: Discussion

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

# THIS POST IS CURRENTLY IN DEVELOPMENT. PLEASE COME BACK SOON!

(

Before we dive in, I want to share a fun little experiment. I used Google NotebookLM to generate a "podcast" from the companion [preprint]( {% link "assets/pdf/Tiered_Transformer.pdf" %}). Take a listen:

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/Tiered_Transformers_enable_parallel_text_generation.m4a" controls=true %}
    </div>
</div>
<div class="caption">
    An AI-generated "podcast" segment on the topic of this blog post.
</div>

)

Think back to your last piece of writing longer than a single paragraph. Did you write it linearly from start to finish? Chances are, _not_.

Whether we’re drafting a paper or a blog post, most of us probably don't sit down and produce a linear stream of consciousness from the first word to the last.

In practice, we’re architects before we’re prose stylists. We start by sketching the skeleton, breaking a monolithic thesis into (semi-)independent yet interrelated sections that we can tackle in isolation. It’s a recursive decomposition: sections dissolve into paragraphs, and paragraphs into sentences, allowing us to build the work semi-independently as the ideas mature.

Among many benefit of working in this way is that (at least in theory) you could deploy the task of writing each section to a different person. If you're writing a book, you can architect the main story, come up with the general outline of say 10 chapters, and then outsource the task of actually writin the chapters to 10 different ghost writers. In this way, you can actually finish your book 10 times as fast!

Large language models (LLMs), by contrast, generate text in a strictly linear, autoregressive fashion: predicting one token at a time, conditioned only on the sequence that has come before it. This sequential nature means LLMs build the “architecture” and the “style” simultaneously, token by token, without the benefit of a high-level blueprint. While this produces impressively coherent output in many cases, it cannot be distributed among parallel workers.


<!-- Now, sometimes a breakthrough in one section forces a refactor of three others, but the core workflow remains the same: a non-linear descent from abstract concepts to concrete words. -->


## Here's an example:

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

>DG inputs regulate CA3 sharp-wave ripples, high-frequency oscillatory events linked to memory consolidation and replay. CA3, the principal downstream target of the dentate gyrus via the mossy fiber pathway, possesses extensive recurrent collateral connections that endow it with autoassociative properties suited to pattern completion and sequence storage. Periods of behavioral quiescence and slow-wave sleep are punctuated by sharp-wave ripples, during which reactivation of previously encoded neuronal sequences is thought to transfer information to neocortical long-term stores. The degree to which the dentate gyrus gates or modulates this consolidation process through its projections to CA3 has become an important question for understanding how memory traces are stabilized over time.

>Beyond sharp-wave ripples, theta oscillations dominate hippocampal dynamics during active exploration and memory encoding. This 6–10 Hz rhythm, sustained by a network encompassing the medial septum and entorhinal cortex, imposes a recurring temporal structure on hippocampal activity that is thought to segregate encoding and retrieval processes across successive cycles. Both the dentate gyrus and CA3 receive prominent theta-modulated excitation from the medial entorhinal cortex, raising the question of how these two external inputs interact and whether they make separable contributions to CA3 network dynamics during behavior.

>During theta, place cells fire in compressed sequential order reflecting the animal's past and future trajectory. As the animal moves through an environment, hippocampal neurons with overlapping place fields discharge in a fixed order within each theta cycle, with each cell advancing toward earlier phases as the animal progresses through its field. The resulting spike sequences, compressed from behavioral timescales into windows compatible with synaptic plasticity, are candidate substrates for the rapid encoding of spatial trajectories. Determining which afferent inputs to CA3 are necessary for generating and maintaining this temporal organization is therefore essential for understanding how the hippocampal circuit constructs and stores sequential representations of experience.