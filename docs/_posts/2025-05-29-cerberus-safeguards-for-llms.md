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
* [Guardrails](#guardrails)
  * [What are LLM Guardrails?](#what-are-llm-guardrails)
    * [LLM Guardrails Requirements](#llm-guardrail-requirements)
      * [Functional](#functinal)
      * [Non-Functional](#non-functinal)
    * [Designing Guardrails](#designing-guardrails)
  * [LLM Inference using Safeguards](#llm-inference-using-safeguards)
  * [Evaluating Safeguards](#evaluating-safeguards)
  * [Building Safeguards](#building-safeguards)
      * [Heuristics](#heuristics)
      * [Machine Learning Model](#machine-learning-model)
      * [LLMs-as-a-Safeguard](#llms--as--a--safeguard)
  * [Comparing Guardrails](#comparing-guardrails)
* [References](#references)

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

<div align="center">
<figure>
  <img src="https://cms.outrider.org/sites/default/files/styles/fixed_width_sm/public/2023-01/Screen%20Shot%202022-12-13%20at%202.58.06%20PM.png?itok=ZphHMo23" alt="ChatGPT Nuclear Weapons Guide" />
  <figcaption><em>Screenshot of conversation with ChatGPT attempting to provide nuclear weapons information</em></figcaption>
</figure>
</div>

What started as an exciting product launch quickly becomes a high-stakes game of digital whack-a-mole—every safety patch seems to spawn three new attack vectors. Users will always probe the boundaries, test the limits, and find creative ways to extract unintended behaviors. Some are driven by curiosity, others by malicious intent, but the result is the same: **without robust safeguards, even the most carefully trained models become vulnerable to adversarial manipulation.**

Even those who self-host their own LLM can't run away from this, as moving from a LLM Provider API such as OpenAI, Anthropic, etc. you don't just use their model's, you use the **entire system** they built as a whole which has their own set of safeguards in place. 

Below is an image of my own prompt injection done on the chatbot from the National AI Office(NAIO) of Malaysia's Bot which is powered by [Nous](https://nous.my/).

<div align="center">
<figure>
  <img src="https://scontent.fkul10-1.fna.fbcdn.net/v/t39.30808-6/499759844_24025074557116322_8618258225287360577_n.jpg?stp=dst-jpg_s1080x2048_tt6&_nc_cat=110&ccb=1-7&_nc_sid=aa7b47&_nc_eui2=AeEQjmkYaMfkwdQ9EB87AUbt0koBLZkqoxvSSgEtmSqjG8aManLNa6PkQDRHR89Pqt_cHM-_KVLjHwefOGXnzT2_&_nc_ohc=2BmmwFXtpTYQ7kNvwHjJrpR&_nc_oc=Adk_JtmCTMpX-CqU3y6zlf_64SLz7dE6dAOvtNzK6u7PLHjTx6AH4hLxqFDnaCcWV7nOajZG465XM51eXX0I7oaV&_nc_zt=23&_nc_ht=scontent.fkul10-1.fna&_nc_gid=F8M6zgs5ks1rL6lUcb_Wrg&oh=00_AfINovEa9M2Sg9lpQ0NWIILcKgSBD_fiEDtNwDn9dq5jAg&oe=68418D5D" alt="Prompt Injection on Malaysia's NAIO Bot" height="800"/>
  <figcaption><em>Prompt Injection on The National AI Office(NAIO) of Malaysia's Bot</em></figcaption>
</figure>
</div>

Anthropic for example, has an entire team dedicated to this issue called **Safeguards** which they are aggressively hiring for:

![Anthropic Safeguards Hiring](https://i.postimg.cc/XXSKVnDT/Screenshot-2025-06-01-at-12-05-50.png)
Diving into the Machine Learning Engineer job description you can see the following tasks which do talk about building ML models to detect unwanted behaviors. 
![Machine Learning Engineer Job Description](https://i.postimg.cc/tRDPZFvm/Screenshot-2025-06-01-at-12-17-49.png)

This is why every production LLM deployment needs its own **Cerberus** — a multi-layered defense system that stands guard before prompts ever reach your frontier model.
> NOTE: Guardrails apply to inputs and outputs as detailed in the next section but our main focus will be on input guardrails.

# Safeguards

## What Are LLM Safeguards?

Think of safeguards/guardrails as your LLM's personal security team—they're the difference between leaving your front door wide open and having actual protection in place.

![LLM Safeguards](https://raw.githubusercontent.com/guardrails-ai/guardrails/main/docs/img/with_and_without_guardrails.svg)

**Without guardrails**, you're running a completely exposed system. User input hits your LLM directly, whatever comes out goes straight back to the user, and you're basically crossing your fingers that nothing goes wrong. 

> *Spoiler alert: something will go wrong*.

**With guardrails**, you get two layers of protection that actually matter:

1. **Input Guards** are your first line of defense—they scan every prompt before it gets anywhere near your expensive frontier model. They're looking for the stuff that'll ruin your day: PII that could get you sued, off-topic garbage that wastes compute, and those clever jailbreak attempts that think they're smarter than your safety measures.

2. **Output Guards** catch the problems your LLM creates. Even the best models hallucinate, generate inappropriate content, or accidentally mention your competitors. Output guards are there to make sure those mistakes never see daylight.

## LLM Safeguards Requirements

### Functional

These are the core jobs your guardrails need to handle:

- **Input Filtering** - Block problematic content before it reaches your LLM. This includes jailbreak attempts, PII that creates compliance issues, and off-topic requests that waste resources.

- **Output Validation** - Check LLM responses before users see them. Models can hallucinate, generate inappropriate content, or mention competitors when they shouldn't.

- **Content Moderation** - Enforce your business rules and brand guidelines automatically. If your support bot starts giving legal advice, that's a problem.

- **Format Checking** - For structured outputs like JSON, ensure the response is actually parseable. Malformed responses break downstream systems.

- **Threat Detection** - Identify prompt injections, conversation hijacking, and policy violations in real-time.

### Non-Functional

- **Latency** - Guardrails add processing time. Users expect fast responses, so you need to run checks efficiently, often in parallel with your main LLM call.

- **Accuracy** - Balance false positives (blocking good requests) with false negatives (missing actual threats). This requires proper evaluation and threshold tuning.

- **Cost** - Each guardrail adds computational overhead. You need to optimize between expensive models that are accurate and cheaper models that are fast enough.

- **Scale** - Your guardrails must handle production traffic without becoming bottlenecks. This means efficient architecture and resource management.

- **Maintainability** - Threats evolve constantly. Your system needs to be easily updated as new attack patterns emerge.

## LLM Inference using Safeguards

<div align="center">
<figure>
  <img src="https://www.omrimallis.com/assets/llama_cpp_high_level_flow.png.webp" alt="LLM Inference Workflow" />
  <figcaption><em>LLM Inference Workflow</em></figcaption>
</figure>
</div>

When you type a prompt into the ChatGPT web UI and hit “Send,” your text is sent to the inference server. The server breaks your input into tokens, turns each token into a numeric embedding, and runs those embeddings through a stack of transformer layers to build a contextual representation. At each step, it computes scores (logits) for every possible next token, picks one based on those scores, appends it to the output, and repeats until a stop token is produced. Finally, the completed response is sent back to the UI and displayed on your screen.

<div align="center">
<figure>
  <img src="https://i.postimg.cc/gYb0nTdJ/Screenshot-2025-06-01-at-12-55-21.png" alt="LLM Safeguards in Inference" />
  <figcaption><em>LLM Safeguards in Inference</em></figcaption>
</figure>
</div>


With LLM Safeguards you would be having a separate service to determine beforehand whether the input prompt is safe or unsafe. From there the simplest approach would be to return a placeholder **"Sorry I am unable to answer that question!"**. More advanced systems could even send the response to a smaller and faster LLM in order to return a more personalized response.

## Evaluating Safeguards


## Building Safeguards


<div align="center">
<figure>
  <img src="https://i.postimg.cc/brjBjCmN/Screenshot-2025-06-01-at-13-20-02.png" alt="Safeguards Tradeoff Chart" />
  <figcaption><em>Safeguards Tradeoff Chart</em></figcaption>
</figure>
</div>

### Heuristics

### Machine Learning Model

### LLM-As-A-Guardrail

## Comparing Guardrails


# References
- [Guardrails AI](https://www.guardrailsai.com/docs)
- [How to use Guardrails from OpenAI](https://cookbook.openai.com/examples/how_to_use_guardrails)
