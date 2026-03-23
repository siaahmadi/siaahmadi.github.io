---
layout: distill
title: Can nine women make a baby in one month? Scaling up autoregressive inference in LLMs
description: We will explore a novel architecture that enables scaling up inference in large language models without sacrificing the autoregressive factorization of the output's joint probability distribution.
tags: ai, optimization, neural-network, llm, systems, transformer
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


# Post in development. Please come back soon!