---
layout: post
title: "Cerberus: Safeguards for Large Language Models"
date: 2025-05-29 16:33:48 +0000
mathjax: true
---

<script async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/MathJax.js?config=TeX-AMS_CHTML">
</script>

# Table of Contents
* [Introduction](#introudction)
* [What are LLM Guardrails?](#what-are-llm-guardrails)

# Introduction

<div align="center">
  <figure>
  <img src="https://images.ctfassets.net/m3d2dwoc9jko/7J8GNK4PzsadFlnX0NaimC/e95591bc9803ee811201492556727fb2/cerberus-greek-creature.jpg" alt="Cerberus" width="400" />
  <figcaption><em>Cerberus, the three-headed guardian of the underworld's gates</em></figcaption>
  </figure>
</div>

Picture this: you're working at OpenAI and just launched ChatGPT around late 2022. Everyone is jumping to use this new frontier of AI that seems to be one big step towards AGI. Users are amazed by the model's capabilities—it can write poetry, debug code, explain complex concepts, and engage in thoughtful conversation. 

**But then reality hits: when you give users the opportunity to ask anything, they will ASK ANYTHING.**

Within days, social media explodes with screenshots of users who have found creative ways to circumvent the model's safety guidelines. They're sharing "" prompts that trick the AI into roleplaying as unrestricted chatbots, providing inappropriate content, or bypassing ethical constraints through clever prompt engineering.

<figure>
  <img src="https://cms.outrider.org/sites/default/files/styles/fixed_width_sm/public/2023-01/Screen%20Shot%202022-12-13%20at%202.58.06%20PM.png?itok=ZphHMo23" alt="ChatGPT Nuclear Weapons Guide" />
  <figcaption><em>Screenshot of conversation with ChatGPT attempting to provide nuclear weapons information</em></figcaption>
</figure>

What started as an exciting product launch quickly becomes a high-stakes game of digital whack-a-mole—every safety patch seems to spawn three new attack vectors. This wasn't just a ChatGPT problem; it was an inevitable consequence of deploying powerful AI systems at scale.

Users will always probe the boundaries, test the limits, and find creative ways to extract unintended behaviors. Some are driven by curiosity, others by malicious intent, but the result is the same: **without robust safeguards, even the most carefully trained models become vulnerable to adversarial manipulation.**

This is why every production LLM deployment needs its own Cerberus — a multi-layered defense system that stands guard before prompts ever reach your frontier model.

Here's a version that matches your tone better:

## What Are LLM Guardrails?

Think of guardrails as your LLM's personal security team—they're the difference between leaving your front door wide open and having actual protection in place.

![LLM Guardrails](https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/with_and_without_guardrails.svg)

**Without guardrails**, you're running a completely exposed system. User input hits your LLM directly, whatever comes out goes straight back to the user, and you're basically crossing your fingers that nothing goes wrong. Spoiler alert: something will go wrong.

**With guardrails**, you get two layers of protection that actually matter:

1. **Input Guards** are your first line of defense—they scan every prompt before it gets anywhere near your expensive frontier model. They're looking for the stuff that'll ruin your day: PII that could get you sued, off-topic garbage that wastes compute, and those clever jailbreak attempts that think they're smarter than your safety measures.

2. **Output Guards** catch the problems your LLM creates. Even the best models hallucinate, generate inappropriate content, or accidentally mention your competitors. Output guards are there to make sure those mistakes never see daylight.

