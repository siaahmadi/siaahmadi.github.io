---
layout: post
title: a post with chat conversations
date: 2024-01-15 10:00:00
description: how to embed iMessage-style conversation threads in al-folio blog posts
tags: formatting
categories: sample-posts
---

The `chat_message` include lets you embed clean, iMessage-style conversation threads directly in your posts — great for showcasing LLM interactions, dialogue examples, interview excerpts, or any back-and-forth exchange.

---

## Basic usage

Wrap one or more `{% raw %}{% include chat_message.html %}{% endraw %}` calls inside a `<div class="chat-container">`:

```html
<div class="chat-container">
  {% raw %}{% include chat_message.html role="user" name="You" content="What is the capital of France?" %}
  {% include chat_message.html role="assistant" name="Claude" content="The capital of France is **Paris**." %}{% endraw %}
</div>
```

Which renders as:

<div class="chat-container">
  {% include chat_message.html role="user" name="You" content="What is the capital of France?" %}
  {% include chat_message.html role="assistant" name="Claude" content="The capital of France is **Paris**." %}
</div>

---

## Multi-turn conversation

<div class="chat-container">
  {% include chat_message.html role="user" name="You" content="Can you explain what a transformer neural network is in plain English?" %}
  {% include chat_message.html role="assistant" name="Claude" content="Sure! Think of a transformer as a very sophisticated attention machine. Instead of reading a sentence word-by-word from left to right (like older models did), it looks at **all the words at once** and figures out which ones are most relevant to each other." %}
  {% include chat_message.html role="user" name="You" content="So it's like reading the whole page before deciding what's important?" %}
  {% include chat_message.html role="assistant" name="Claude" content="Exactly — that's a great analogy. The \"attention\" mechanism lets every word vote on how much it should pay attention to every other word in the sentence. That's why transformers are so good at capturing long-range dependencies in text." %}
</div>

---

## With timestamps

Pass a `time` parameter to show a timestamp beneath each bubble:

<div class="chat-container">
  {% include chat_message.html role="user"      name="Alice" content="Are you free for lunch on Friday?" time="10:14 AM" %}
  {% include chat_message.html role="assistant" name="Bob"   content="Yes! Where were you thinking?" time="10:16 AM" %}
  {% include chat_message.html role="user"      name="Alice" content="That new ramen place on 5th?" time="10:17 AM" %}
  {% include chat_message.html role="assistant" name="Bob"   content="Perfect, see you at noon 🍜" time="10:18 AM" %}
</div>

---

## Without author names

Omit `name` for a cleaner iMessage-style look:

<div class="chat-container">
  {% include chat_message.html role="user"      content="Remind me: what did we decide about the API design?" %}
  {% include chat_message.html role="assistant" content="We agreed to use REST with JWT authentication for v1, and revisit GraphQL once the team has grown." %}
  {% include chat_message.html role="user"      content="Right, thanks." %}
</div>

---

## Using the divider

Use `<div class="chat-divider">Label</div>` to separate sections of a conversation:

<div class="chat-container">
  {% include chat_message.html role="user"      name="You"    content="Can you write a haiku about the ocean?" %}
  {% include chat_message.html role="assistant" name="Claude" content="Waves crash endlessly,\nSalt and foam kiss the cold shore,\nThe deep holds its breath." %}
  <div class="chat-divider">Later that day</div>
  {% include chat_message.html role="user"      name="You"    content="Now one about mountains?" %}
  {% include chat_message.html role="assistant" name="Claude" content="Stone older than words,\nClouds break against the summit,\nSilence fills the gaps." %}
</div>

---

## Markdown inside bubbles

Message content is passed through the Markdown processor, so you can use **bold**, _italic_, `inline code`, and even links:

<div class="chat-container">
  {% include chat_message.html role="user" name="You" content="What's the Python syntax for a list comprehension?" %}
  {% include chat_message.html role="assistant" name="Claude" content="The basic syntax is `[expression for item in iterable]`. For example, `[x**2 for x in range(10)]` gives you the squares of 0–9. You can also add a filter: `[x for x in range(10) if x % 2 == 0]`." %}
</div>

---

## Include parameters reference

| Parameter | Required | Values | Description |
|-----------|----------|--------|-------------|
| `role`    | ✓ | `"user"` / `"assistant"` | Controls which side the bubble appears on |
| `content` | ✓ | any string (Markdown OK) | The message text |
| `name`    | – | any string | Author label above the bubble |
| `time`    | – | any string | Timestamp below the bubble |

The `role` parameter is flexible: `"user"` and `"right"` both produce a right-aligned bubble; everything else (including `"assistant"`, `"bot"`, `"left"`, or any custom string) produces a left-aligned bubble.
