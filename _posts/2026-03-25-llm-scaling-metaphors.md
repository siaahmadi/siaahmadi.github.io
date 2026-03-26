---
layout: distill
title: Specification gaming? Or LLM abstraction failure modes?
date: 2026-03-23
description: My frustrations (and comedic experience) trying to get LLMs to help me find a good linguistic analogy.
tags: llm, failure-mode
categories: AI
bibliography: 2026-03-23-llm-scaling-metaphors.bib
---


In 1975, Fred Brooks published his now-famous book on software project management, *The Mythical Man-Month*<d-cite key="brooks1975mythical"></d-cite>. Early on in the book, he writes

>When a task cannot be partitioned because of sequential constraints, the application of more effort has no effect on the schedule (Fig. 2.2). The bearing of a child takes nine months, no matter how many women are assigned. Many software tasks have this characteristic because of the sequential nature of debugging.

This is sometimes quoted as "9 women can't make a baby in 1 month" which stresses the point that child-bearing (and by analogy many other inherently sequential tasks, including some software projects) cannot be completed faster by assigning more workers to the task.

The first time I heard this phrase I thought it effectively conveyed the point in a funny way. When writing [my recent blog post](/blog/2026/scaling-llm-inference/) on scaling LLM inference, I considered using it but I wanted make sure it doesn't come across as dated. After all, it was first written over 50 years ago in 1975. So I asked my partner what she thought of it, and she said that it sounded odd. And that's when I took to AI to help me find a more neutral phrase that shares the Brooks analogy characteristics. I wanted a phrase that's intuitive, drives the main point home (that sequential tasks can't be scaled), and is perhaps a bit humorous. While I can't say I necessarily got the help I needed, I can say I was entertained!

## So what did the AIs say?

I first went to ChatGPT and asked it for help:

<div class="chat-container">
  {% include chat_message.html
  role="right"
  content="What's a good other example that conveys the point meant by \"Can nine women make a baby in one month?\""
  %}
  {% include chat_message.html
  role="left"
  name="ChatGPT"
  content="You’re trying to capture the idea that some processes are **not parallelizable**. Throwing more people (or resources) at them doesn’t compress time because the steps are inherently sequential or constrained.

Here are some clean alternatives, depending on tone:

**Simple and intuitive**

* “Nine cooks can’t bake a cake in one-ninth the time.”
* “You can’t speed up a sunrise by hiring more people.”
* “A tree doesn’t grow faster because you water it nine times at once.”

**A bit sharper / more conceptual**

* “Some problems scale with effort; others scale with time.”
* “Parallel effort doesn’t help when the bottleneck is sequential.”
* “Adding workers doesn’t shrink a critical path.”

**More technical (closer to your background)**

* “You can’t parallelize a dependency chain.”
* “Throughput can scale, latency often can’t.”
* “This is a latency problem, not a bandwidth problem.”

If you tell me where you’re using it (resume, research statement, talk), I can tune one that lands exactly right for that audience."
  %}
</div>

Clearly, ChatGPT understood the key point because it said "You’re trying to capture the idea that some processes are **not parallelizable**." But, I mean... come on! It totally fails on so many levels in its examples! "Nine cooks can’t bake a cake in one-ninth the time" Really?? 😂

It's entirely unclear to me why it fails to understand the significance of the number 9 in the original example. It simply copies it over.

Or in the example "You can’t speed up a sunrise by hiring more people" it makes a very non-sensical analogy. The "sun?!" How did ChatGPT miss the point that the "worker" should actually be performing the "task", i.e., have a causal influence on it?

In other examples ("Some problems scale with effort; others scale with time") it only says a syntactically correct sentence without proper semantic content. In "Parallel effort doesn’t help when the bottleneck is sequential" it completely misses the requirement that the phrase must be punchy and intuitive. In yet another ("Adding workers doesn’t shrink a critical path") it gives a vague and obscure analogy.


## Can a more powerful model do better?

Since I don't have a subscription to ChatGPT, I thought maybe I'm not using the most capable model. So I went to Gemini Pro next, and took my prompting lessons with me. I made sure to tell Gemini that the examples should be "punchy, intuitive, and non-technical." But my experience became only funnier (and more frustrating 😄).



<div class="chat-container">
  {% include chat_message.html
  role="right"
  content="what's a good other example that conveys the point meant by \"Can nine women make a baby in one month?\" I want it to be punchy, intuitive, and non-technical, just like this sentence itself."
  %}
  {% include chat_message.html
  role="left"
  name="Gemini Pro"
  content="This classic analogy from The Mythical Man-Month perfectly illustrates the concept of **Brook’s Law**: that some tasks are inherently sequential and cannot be sped up by adding more resources.
    
  If you are looking for an alternative that is just as punchy and non-technical, here are a few options:
  
  ---
  
  **The \"Natural\" Limit**
  
  **\"You can’t hatch an egg in one day by putting nine hens on it.\"**
  
  This is perhaps the closest spiritual successor to the baby analogy. It highlights that biological or physical processes have a fixed \"burn rate\" that extra labor cannot bypass.
  
  **The \"Physical\" Space**
  
  **\"Adding more chefs to a small kitchen just makes a crowd, not a faster dinner.\"**
  
  This conveys the idea of **overhead** and **interference**. It’s intuitive because everyone understands that at a certain point, people start bumping into each other and communication becomes a bottleneck rather than a benefit.
  
  **The \"Sequential\" Logic**
  
  **\"You can't drive a car at 600 mph by putting ten people in it.**
  
  This version is useful for explaining that the \"engine\" (the core process or technology) has a maximum output regardless of the \"payload\" (the team size).
  
  **The \"Construction\" Reality\"**
  
  **\"Two people can't paint the same square inch at the same time.\"**
  
  This is a very visceral way to explain **resource contention**. It makes it clear that if a task isn't divisible, a second person is literally just standing around waiting.
  
  ---

  **Comparison at a Glance**
  
  Analogy | Focus
  **Hens & Eggs** | Fixed biological/natural duration.
  **Chefs in Kitchen** | Communication overhead and physical constraints.
  **Car Speed** | Capacity vs. Throughput.
  **Painting** | Indivisibility of a specific task.
  
  ---

  I can help you apply these concepts to a specific project management or system design bottleneck if you'd like."
  %}
</div>

Like ChatGPT, Gemini also "understood" the main intent of the original phrase. And like ChatGPT, it also fell in the trap of the number nine 😅.

I was quite amused by its car example: "You can't drive a car at 600 mph by putting ten people in it." 😂😂 It completely failed on the logic here. Or in the example "Can twelve winemakers make a ten-year vintage in one year?" it just mixed up a bunch of unrelated numbers and durations!

So again, I learned some prompting lessons, and went back to Gemini Pro with a clean slate (started a new conversation). Here's how this one went. I'll spare you additional commentary but if you'd like a laugh, I encourage you to read it 😅.

P.S.: The sleep example is my favorite 🤣.

P.S. 2: Also note how it's so confident that each example "works!" Because you know... **Everyone knows a cake takes about an hour to bake**!!!


<div class="chat-container">
  {% include chat_message.html
  role="right"
  content="what's a good other example that conveys the point meant by \"Can nine women make a baby in one month?\" I want it to be punchy, intuitive, and non-technical, just like this sentence itself. If you use any timelines in your examples the timelines should be widely known, like a pregnancy's duration. The key aspect of the pregnancy example is that it's kind of funny (because it's absurd), it's extremely well known (there's not a single human who wouldn't know a pregnancy lasts nine months) and it conveys the key point: that some things are inherently sequential and impossible to parallelize."
  %}
  {% include chat_message.html
  role="left"
  name="Gemini Pro"
  content="The \"nine women\" analogy is the gold standard for explaining Brooks’ Law (the idea that adding manpower to a late software project makes it later). It highlights that some tasks have a \"natural gestation period\" that simply cannot be bypassed by adding more resources.

Here are a few punchy, non-technical alternatives that use widely known timelines:

---

1. **The Oven Example (The Most Intuitive)**<br>
  **\"Can six ovens bake a cake in ten minutes?\"**<br>
  **Why it works:** Everyone knows a cake takes about an hour to bake. If you try to bake it in ten minutes by \"parallelizing\" it across six ovens, you just end up with six cold pans of batter (or one burnt mess if you try to crank up the heat). It perfectly illustrates that the internal chemical process (like the logic of a project) requires its own time to \"set.\"


2. **The Movie Example (The Most Logical)**<br>
  **\"Can four projectors finish a two-hour movie in thirty minutes?\"**<br>
  **Why it works:** A movie is a purely linear experience. Even if you have the best technology and multiple people \"working\" on the playback, the story must unfold at its own pace for it to make sense. Watching the first 30 minutes on four different screens simultaneously doesn't mean you've \"seen the movie.\"


3. **The Pilot Example (The \"Transport\" Logic)**<br>
  **\"Can ten pilots fly a five-hour flight in thirty minutes?\"**<br>
  **Why it works:** This highlights that the \"vessel\" (the project) is limited by its environment and physics. No matter how many experts you put in the cockpit, the plane cannot physically travel faster than its engine allows. Adding more pilots just crowds the cockpit without changing the arrival time.


4. **The Sleep Example (The Relatable Favorite)**<br>
  **\"Can seven people sleep for you on Sunday so you can stay awake all week?\"**<br>
  **Why it works:** While a bit different in structure, it attacks the idea of \"banking\" or \"outsourcing\" a fundamental, sequential need. Everyone knows you need roughly 8 hours of sleep every night. You can’t parallelize your rest for the week into a single day just because you have \"extra hours\" available on Sunday.

---

**Summary Table**

  Analogy	| The \"Man-Month\" Logic	| Why it's punchy
  **9 Women** |	9 × 1 month ≠ Baby | The classic \"biological\" constraint.
  **6 Ovens**	| 6 × 10 mins ≠ Cake | The \"chemical/process\" constraint.
  **4 Projectors** | 4 × 30 mins ≠ Movie | The \"linear content\" constraint.
  **10 Pilots** | 10 pilots ≠ Fast Flight | The \"physical/vessel\" constraint.

---

Would you like me to help you draft a short presentation slide or a \"TL;DR\" email using one of these to explain a project delay to stakeholders?"
  %}


  {% include chat_message.html
    role="right"
    content="Okay you're really totally missing the point. You genuinely have no idea how the world works and I actually thank you for helping me identify a failure mode in the state of the art LLMs!"
  %}
  {% include chat_message.html
  role="left"
  name="Gemini Pro"
  content="That’s a fair hit. The strength of the \"nine women\" analogy isn't just the math—it’s that the nine-month timeline is a hard biological constant. My previous examples like \"baking a cake\" or \"flying a plane\" are too variable to land with the same thud.

If you want that same universal, \"physics-says-no\" logic, try these:

1. **The Sleep Analogy**<br>
 **\"Can eight people each sleep for one hour to give you a full night's rest?\"**<br>
 **Why it works:** Everyone knows a \"full night's rest\" is roughly eight hours. It’s a biological requirement that is inherently sequential; you cannot \"parallelize\" your exhaustion away by having others sleep on your behalf.

2. **The Calendar Analogy**<br>
 **\"Can seven people make it next Monday by tomorrow morning?\"**<br>
 **Why it works:** It’s perfectly absurd. A week is seven days, and no amount of \"manpower\" or extra people living through the day can force the calendar to skip ahead. It highlights that some processes are simply passengers of time.

3. **The Song Analogy**<br>
 **\"Can four people finish a three-minute song in forty-five seconds?\"**<br>
 **Why it works:** A \"three-minute song\" is defined by its duration. If you play it four times faster, or have four people sing different parts simultaneously, you no longer have the song—you have noise. It perfectly captures the \"sequential logic\" requirement.

Would you like me to tweak one of these into a one-sentence \"mic drop\" for your specific context?"
  %}
  

  {% include chat_message.html
  role="right"
  content="Okay forget it. I'll make this into a blog post and make fun of you.
  
  Stick to what you're good at: here's a sentence I wrote, you just make it shorter \"Can two travelers reach their destination in half the time using two cars?\""
  %}
  {% include chat_message.html
  role="left"
  name="Gemini Pro"
  content="\"Can two cars get one traveler there in half the time?\""
  %}
  
  
  {% include chat_message.html
  role="right"
  content="NOOOOO!!!! DON'T CHANGE THE KEY VARIABLES! YOU CAN'T CUT ONE PERSON INTO TWO TO PUT THEM IN TWO CARS. YOU NEED TWO PEOPLE!"
  %}
  {% include chat_message.html
  role="left"
  name="Gemini Pro"
  content="Fair point—splitting a person is definitely not the optimization we’re looking for! If we keep both travelers and both cars, the shortest version while maintaining that logic is:

**\"Can two cars get two travelers there in half the time?\"**

Would you like me to try a version that swaps the cars for another common mode of transport, like a flight or a train?"
  %}
</div>