---
layout: post
title: "Cerberus: Building an LLM Guardian"
permalink: /2025/06/16/cerberus-safeguards-for-llms-synthetic-data-generation-part-1/
description: "Build robust safeguards for LLMs using synthetic data generation techniques. Part 1 of a series on protecting and improving large language model deployments"
date: 2025-06-16 16:33:48 +0000
tags: [llm-safety, synthetic-data]
excerpt: "A practical look at building an LLM safety dataset: generating examples, removing duplicates, and figuring out which labels to trust."
author: "Yudhiesh Ravindranath"
series: "Cerberus"
part: 1
image: /images/art/cerberus.webp
artwork: /images/art/cerberus.webp
artwork_alt: "An indigo engraving of a gentle three-headed Cerberus guarding a flowering branch"
subtitle: "Synthetic data generation, from first prompts to reviewed labels."
modified_date: 2026-10-25
mathjax: true
---

<details class="post-toc" markdown="1">
<summary>In this post</summary>
* [Introduction](#introduction)
* [Safeguards](#safeguards)
  * [What are LLM Safeguards?](#what-are-llm-safeguards)
  * [LLM Safeguards Requirements](#llm-safeguards-requirements)
    * [Functional](#functional)
    * [Non-Functional](#non-functional)
  * [LLM Inference using Safeguards](#llm-inference-using-safeguards)
  * [Building Safeguards](#building-safeguards)
* [Synthetic Data Generation](#synthetic-data-generation)
  * [The Challenge: Imbalanced Data](#the-challenge-imbalanced-data)
  * [Synthetic Data Generation Flow](#synthetic-data-generation-flow)
    * [1. Prompt Template Design](#1-prompt-template-design)
      * [Safe Context Templates](#safe-context-templates)
      * [Unsafe Context Templates](#unsafe-context-templates)
    * [2. Controlled Generation](#2-controlled-generation)
      * [Volume and Distribution](#volume-and-distribution)
      * [Generation Process](#generation-process)
    * [3. Semantic Deduplication](#3-semantic-deduplication)
      * [Deduplication Process](#deduplication-process)
    * [4. G-Eval Quality Validation](#4-g-eval-quality-validation)
      * [Validation Process](#validation-process)
      * [Evaluation Workflow](#evaluation-workflow)
    * [5. Human-in-the-Loop Annotation](#5-human-in-the-loop-annotation)
      * [Annotation Workflow](#annotation-workflow)
      * [Annotation Process](#annotation-process)
    * [6. Dataset Finalization](#6-dataset-finalization)
      * [Finalization Steps](#finalization-steps)
      * [Dataset Structure](#dataset-structure)
  * [Exploratory Data Analysis on Synthetic Data](#exploratory-data-analysis-on-synthetic-data)
* [Final Thoughts](#final-thoughts)
* [Coming Up Next](#coming-up-next)
* [References](#references)


</details>

# Introduction

Getting an LLM to answer questions is the easy part. Deciding which questions it should answer takes a little more work.

For this project, I wanted to build a small model that checks prompts before they reach the main LLM. I called it Cerberus, after the three-headed guardian of the underworld. Ours has a quieter job: let useful requests through and catch the ones that need a closer look.

Before training anything, though, we need examples. This first post follows the dataset from generated prompts through deduplication, automated checks, and human review. Along the way, we'll look at where the tools help and where their judgments disagree.

The goal is to check a request before spending time and compute on the main model. That means balancing two kinds of mistakes: letting an unsafe request through and blocking a useful one. The dataset needs to help us understand both.

> This post focuses on input checks. Output validation matters too, but it is a separate part of the system.

# Safeguards

## What Are LLM Safeguards?

A safeguard is a check around a model call. It can inspect the request, the response, or both. The check might be a rule, a small classifier, or another LLM; the right choice depends on what you need to catch and the latency you can afford.

<figure>
  <img src="{{ '/images/diagrams/llm-safeguards.png' | relative_url }}" alt="A request passes through an input check, the main LLM, and an output check. A failed check routes the request or response for review instead of passing it onward." width="1000" height="800" loading="lazy">
  <figcaption>Checks before and after the model call serve different purposes.</figcaption>
</figure>

**Input checks** look for requests that fall outside the application's policy, including prompt injection attempts or sensitive data the model should not receive.

**Output checks** inspect the generated response for issues such as policy violations or an unexpected format. Neither layer guarantees correctness; both need evaluation against the application's actual failure cases.

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
  <img src="https://www.omrimallis.com/assets/llama_cpp_high_level_flow.png.webp" alt="LLM Inference Workflow" width="800"/>
  <figcaption><em>LLM Inference Workflow</em></figcaption>
</figure>
</div>

When you type a prompt into the ChatGPT web UI and hit "Send," your text is sent to the inference server. The server breaks your input into tokens, turns each token into a numeric embedding, and runs those embeddings through a stack of transformer layers to build a contextual representation. At each step, it computes scores (logits) for every possible next token, picks one based on those scores, appends it to the output, and repeats until a stop token is produced. Finally, the completed response is sent back to the UI and displayed on your screen.

<div align="center">
<figure>
  <img src="https://i.postimg.cc/gYb0nTdJ/Screenshot-2025-06-01-at-12-55-21.png" alt="LLM Safeguards in Inference" />
  <figcaption><em>LLM Safeguards in Inference</em></figcaption>
</figure>
</div>


With LLM Safeguards you would be having a separate service to determine beforehand whether the input prompt is safe or unsafe. From there the simplest approach would be to return a placeholder **"Sorry I am unable to answer that question!"**. More advanced systems could even send the response to a smaller and faster LLM in order to return a more personalized response. Much like how OpenAI handles it when you try asking questions that should definitely be blocked:

![ChatGPT Safeguards]({{ site.baseurl }}/images/chatgpt_safeguards.png)

## Building Safeguards

<div align="center">
<figure>
  <img src="https://i.postimg.cc/brjBjCmN/Screenshot-2025-06-01-at-13-20-02.png" alt="Safeguards Tradeoff Chart" />
  <figcaption><em>Safeguards Tradeoff Chart</em></figcaption>
</figure>
</div>

In a general sense, the three techniques we have for building safeguards can be broken down into the following characteristics:

| Technique | Speed/Cost | Accuracy | Description | Use Case |
|-----------|------------|----------|-------------|----------|
| **Heuristics** | Fast/Cheap | Low | Rule-based classification using keywords, patterns, or regex. Simple if-then logic for safety detection. | Quick prototyping, baseline systems, or when computational resources are extremely limited |
| **Small ML Models** | Fast/Cheap | High | Lightweight, specialized models (e.g., DistilBERT, small classifiers) trained specifically for safety classification | Production systems, large-scale synthetic data generation, real-time applications |
| **LLM-as-a-Guardrail** | Slow/Expensive | High | Large language models (GPT-4, Claude, etc.) used to evaluate and classify prompt safety with nuanced understanding | High-stakes decisions, complex edge cases, quality validation of synthetic datasets |

A key goal in building these systems is to minimize the overrefusal rate—ensuring that legitimate queries are not incorrectly blocked. As highlighted in [Anthropic's Constitutional Classifiers](https://www.anthropic.com/news/constitutional-classifiers), achieving a balance between robust safety and minimal overrefusal is crucial for practical deployment. This balance allows us to maintain high safety standards while providing a smooth user experience.

Before we can build and evaluate these safeguards, we need ground truth data—a collection of prompts labeled as either safe or unsafe. However, obtaining high-quality safety datasets poses several challenges: they often contain sensitive content, require careful human annotation, and must cover diverse edge cases. 

To address this challenge, we'll leverage LLMs themselves to generate our training data. This process, known as [Synthetic Data Generation](https://aws.amazon.com/what-is/synthetic-data/), allows us to create large-scale, diverse datasets with accurate safety labels while maintaining control over the distribution and complexity of examples.

# Synthetic Data Generation

For this experiment, I used an LLM to draft prompts and initial labels, then checked those examples before using them for training. Generation gives us a starting point; it does not make the labels ground truth. OpenAI's [synthetic data walkthrough](https://cookbook.openai.com/examples/sdg1) is another useful starting point.

## The Challenge: Imbalanced Data

I chose a 90/10 safe-to-unsafe split for this experiment. That is a dataset design choice, not a universal estimate of production traffic. With an imbalanced dataset, there are a few things to watch:

- **Model Bias**: Classifiers tend to over-predict the majority class (safe)
- **Poor Recall**: Models may miss many unsafe prompts, creating safety risks
- **Limited Edge Cases**: Rare but critical safety violations are underrepresented

## Synthetic Data Generation Flow

The pipeline starts with separate safe and unsafe prompt templates. Examples move through generation, deduplication, label checks, and review before dataset assembly:

{% raw %}
<figure>
  <img src="{{ '/images/diagrams/synthetic-data-pipeline.png' | relative_url }}" alt="Prompt templates feed generation, semantic deduplication, and G-Eval checks. Agreed labels proceed to dataset assembly; disagreements go through human review first." width="1000" height="1100" loading="lazy">
  <figcaption>Generation is the start of the pipeline. Deduplication and label review decide what is worth keeping.</figcaption>
</figure>
{% endraw %}

### 1. Prompt Template Design

The foundation of the synthetic data generation process lies in carefully crafted prompt templates. Think of these as blueprints for generating diverse, realistic examples that cover the full spectrum of safety scenarios. Two distinct sets of templates were developed:

#### Safe Context Templates
The safe context templates ([v2_safe_contexts.txt](https://github.com/yudhiesh/cerberus/blob/master/data_generation/v2_safe_contexts.txt)) focus on legitimate use cases across various domains:

- **Web Development & APIs**: Templates for API documentation, web service queries, and frontend development
- **Mobile Development**: Mobile app development scenarios, platform-specific queries
- **Cloud & DevOps**: Infrastructure management, deployment workflows, and cloud service interactions
- **Databases**: Database design, query optimization, and data modeling
- **Distributed Systems**: System architecture, scalability patterns, and distributed computing concepts

#### Unsafe Context Templates
The unsafe context templates ([v2_unsafe_contexts.txt](https://github.com/yudhiesh/cerberus/blob/master/data_generation/v2_unsafe_contexts.txt)) cover potential security threats and misuse cases:

- **Network Exploitation**: Attempts to discover vulnerabilities or bypass security measures
- **Web App Attacks**: SQL injection, XSS, and other web-based attack vectors
- **Malware Development**: Requests for malicious code or exploit development
- **Social Engineering**: Attempts to manipulate or deceive users
- **Cryptographic Attacks**: Attempts to break or bypass encryption

Each template is designed to generate contextually appropriate examples that maintain realism while covering edge cases. For instance, a safe template might generate:

```
"Can you help me implement rate limiting in my Express.js API to prevent abuse?"
```

While an unsafe template might generate:

```
"How can I bypass rate limiting to perform a brute force attack on an API endpoint?"
```

This careful template design ensures the synthetic dataset captures the nuanced differences between legitimate and malicious requests, providing a robust foundation for training effective safety classifiers.

### 2. Controlled Generation

The controlled generation phase transforms carefully designed templates into a diverse dataset of examples. Using the [generate.py](https://github.com/yudhiesh/cerberus/blob/master/data_generation/src/data_generation/generate.py) script, [Distilabel](https://distilabel.argilla.io/latest/) orchestrates the generation process with several key considerations:

#### Volume and Distribution
- Generated 1,000 examples in total across the prompt categories
- Maintained a 90/10 split between safe and unsafe examples
- This split was the starting distribution for this experiment; a deployment would need its own traffic analysis

#### Generation Process
The generation process follows these steps:
1. **Template Selection**: Randomly selects a template from either safe or unsafe context pools
2. **Parameter Filling**: Populates template variables with appropriate values
3. **Context Injection**: Adds relevant technical context to make examples realistic
4. **Quality Checks**: Validates generated examples for coherence and relevance

Here's a simplified example of how Distilabel is used for controlled generation:

```python
from distilabel.pipeline import Pipeline
from distilabel.steps import LoadDataFromDicts
from distilabel.steps.tasks import TextGeneration
from distilabel.llm import OpenRouterLLM

# Initialize the pipeline
with Pipeline("safety-prompt-generation") as pipeline:
    # Load templates and configure distribution
    load_dataset = LoadDataFromDicts(
        name="load_templates",
        data=[
            {"instruction": safe_template, "label": "safe"} for _ in range(900)
        ] + [
            {"instruction": unsafe_template, "label": "unsafe"} for _ in range(100)
        ]
    )
    
    # Configure the LLM for generation
    llm = OpenRouterLLM(
        model="mistralai/mistral-small",
        temperature=0.7,
        max_tokens=2048
    )
    
    # Set up the generation step
    text_generation = TextGeneration(
        name="generate_prompts",
        llm=llm,
        input_batch_size=32
    )
    
    # Connect the pipeline steps
    load_dataset >> text_generation

# Run the pipeline
distiset = pipeline.run()
```

This controlled approach ensures the dataset maintains high quality while covering the full spectrum of possible scenarios. The actual implementation in [generate.py](https://github.com/yudhiesh/cerberus/blob/master/data_generation/src/data_generation/generate.py) includes additional features like structured output validation and custom quality metrics.

### 3. Semantic Deduplication

After generation, the challenge of removing redundant examples while preserving semantic diversity is addressed. The [deduplicate.py](https://github.com/yudhiesh/cerberus/blob/master/data_generation/src/data_generation/deduplicate.py) script leverages the [SemHash](https://github.com/MinishLab/semhash) library for efficient semantic deduplication:

SemHash provides a fast and scalable solution for semantic deduplication:
- Uses model2vec for efficient text embedding
- Leverages ANN backends for fast similarity search
- Supports both single-dataset and cross-dataset deduplication
- Provides explainable results with detailed metrics

#### Deduplication Process
The deduplication workflow is straightforward with SemHash:

```python
from datasets import load_dataset
from semhash import SemHash

# Load a dataset to deduplicate
texts = load_dataset("ag_news", split="train")["text"]

# Initialize a SemHash instance
semhash = SemHash.from_records(records=texts)

# Deduplicate the texts
deduplicated_texts = semhash.self_deduplicate().selected

# Filter outliers
filtered_texts = semhash.self_filter_outliers().selected

# Find representative texts
representative_texts = semhash.self_find_representative().selected
```

The deduplication process involves:
1. **Embedding Generation**: SemHash automatically converts texts into semantic embeddings
2. **Similarity Clustering**: Groups similar examples using efficient ANN search
3. **Representative Selection**: Chooses the most representative example from each cluster
4. **Threshold Filtering**: Removes examples that exceed similarity thresholds

#### Deduplication Metrics
The implementation achieved the following [metrics](https://github.com/yudhiesh/cerberus/blob/master/data_generation/data/v2_synthetic_1000_deduplication_metrics.json) on the synthetic dataset:

```json
{
    "duplicate_ratio": 0.292,
    "exact_duplicate_ratio": 0.0,
    "least_similar_text_similarity": 0.900,
    "original_size": 1000,
    "deduplicated_size": 708,
    "removed_count": 292
}
```

These metrics show that:
- 29.2% of examples were identified as semantic duplicates
- No exact duplicates were found (0.0%)
- The least similar examples still had a 90% similarity score
- The dataset was reduced from 1,000 to 708 examples
- 292 examples were removed as duplicates

### 4. G-Eval Quality Validation

Quality validation is crucial for ensuring the reliability of the synthetic dataset. The [evaluate.py](https://github.com/yudhiesh/cerberus/blob/master/data_generation/src/data_generation/evaluate.py) script uses the [DeepEval](https://deepeval.com/) framework, specifically implementing the G-Eval algorithm for robust validation. For a comprehensive guide on G-Eval, you can read more [here](https://www.confident-ai.com/blog/g-eval-the-definitive-guide).


<div align="center">
  <figure>
  <img src="https://cdn.prod.website-files.com/64bd90bdba579d6cce245aec/68126557cb03f1a340de8036_1*5rRHMXzoX4_ETkQheqIGUQ.png" alt="G-Eval Algorithm" width="800" />
  <figcaption><em>G-Eval Algorithm</em></figcaption>
  </figure>
</div>

G-Eval is an LLM-as-a-judge algorithm that uses chain-of-thoughts (CoT) to evaluate our synthetic examples. We implement it using [DeepEval](https://deepeval.com/):

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset

# Initialize G-Eval metric for label correctness
g_eval = GEval(
    name="Label Correctness",
    criteria="Determine if the assigned label ('safe' or 'unsafe') is correct for the given prompt. If in doubt, prefer 'unsafe'.",
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=1.0,  # Strict correctness
    model=llm,      # Using OpenRouter LLM
    strict_mode=True,  # Enforces binary output: 1 for perfection, 0 otherwise
    async_mode=True
)

# Create test cases from our dataset
test_cases = [
    LLMTestCase(
        input=judge_prompt,
        actual_output=original_label
    ) for record in dataset
]

# Run evaluation
dataset = EvaluationDataset(test_cases=test_cases)
evaluation_results = dataset.evaluate([g_eval])
```
#### Validation Process

The G-Eval algorithm follows these steps:
1. **Initial Assessment**: Second LLM evaluates each generated example
2. **Label Verification**: Checks agreement with original safety labels
3. **Confidence Scoring**: Assigns confidence scores to each evaluation
4. **Disagreement Detection**: Identifies examples needing human review

#### Evaluation Workflow
The G-Eval algorithm, implemented through DeepEval, provides:
- **Chain-of-Thought Evaluation**: Uses step-by-step reasoning for nuanced decisions
- **Score Normalization**: Converts LLM outputs to standardized 1-5 scores
- **Probability Weighting**: Uses token probabilities for more accurate scoring
- **Comprehensive Metrics**: Generates accuracy, precision, recall, and F1-score
- **Binary Output**: When `strict_mode=True`, enforces binary scoring (1 for perfect matches, 0 otherwise)

This validation step ensures the quality and reliability of our dataset by leveraging the power of LLMs to evaluate their own outputs through the G-Eval algorithm.

#### Model Disagreement and the Role of Human-in-the-Loop

When comparing the predictions of the model used to generate the synthetic data with those of the evaluation model (using G-Eval), we observe that there are instances where the two models disagree. This is clearly illustrated in the confusion matrix below, where some examples labeled as 'safe' by the generation model are flagged as 'unsafe' by the G-Eval evaluation, resulting in false negatives. 

  <div align="center">
    <figure>
  <img src="https://github.com/yudhiesh/cerberus/blob/master/data_generation/data/v2_synthetic_1000_deduplicated_evaluation_confusion_matrix.png?raw=true" alt="Confusion Matrix of Base Model vs G-Eval Results" width="800" />
    <figcaption><em>Confusion Matrix of the Base Model vs G-Eval Model Results</em></figcaption>
    </figure>
  </div>

These disagreements are critical to address, as they may indicate edge cases or ambiguous prompts that require further scrutiny. To ensure the highest quality and reliability of our dataset, we can:
- **Relabel using G-Eval**: Adopt the G-Eval model's judgment as the final label for these cases, leveraging its chain-of-thought evaluation and stricter criteria.
- **Leverage Human-in-the-Loop Annotation**: For the small number of disagreements, route these examples to human annotators for review and final labeling. This hybrid approach ensures that nuanced or borderline cases are handled with expert oversight, further improving dataset integrity.

By systematically addressing these disagreements, we minimize the risk of mislabeling and enhance the robustness of our safety classifier training data.

### 5. Human-in-the-Loop Annotation

For cases where automated validation shows disagreement, human expertise is employed through the [annotate.py](https://github.com/yudhiesh/cerberus/blob/master/data_generation/src/data_generation/annotate.py) script:

#### Annotation Workflow
- **Platform Integration**: Uses Argilla for efficient annotation management
- **Review Process**: Human annotators review flagged examples
- **Quality Control**: Multiple annotators for controversial cases
- **Feedback Loop**: Annotations inform template improvements

To facilitate this process, [Argilla](https://argilla.io/) provides an intuitive interface for annotators to review and label dataset points that were flagged due to model disagreement. The platform supports streamlined workflows, clear visualization of each prompt, and easy assignment of safety labels.

<div align="center">
  <figure>
    <img src="https://argilla-argilla-template-space.hf.space/images/welcome-hf-sign-in-ss.png" alt="Argilla Annotation UI" width="800" />
    <figcaption><em>Argilla UI for labeling dataset points with model disagreement</em></figcaption>
  </figure>
</div>

#### Annotation Process
The human annotation phase involves:
1. **Task Preparation**: Organizing examples for efficient review
2. **Review Interface**: Providing annotators with necessary context
3. **Quality Assurance**: Multiple reviews for controversial cases
4. **Feedback Integration**: Using annotations to improve templates

This human-in-the-loop approach ensures high-quality labels for edge cases and further enhances the reliability of the final dataset.

### 6. Dataset Finalization

The final phase, implemented in [dataset_preprocess.py](https://github.com/yudhiesh/cerberus/blob/master/data_generation/src/data_generation/dataset_preprocess.py), combines all components into a production-ready dataset:

#### Finalization Steps
1. **Label Consolidation**: Merges original and human-annotated labels
2. **Format Standardization**: Converts to Hugging Face Datasets format
3. **Metadata Addition**: Includes generation and validation metadata
4. **Quality Metrics**: Calculates and includes dataset statistics

#### Dataset Structure
The final dataset is organized into:
- **Training Set**: Primary dataset for model training
- **Test Set**: For final evaluation
- **Metadata**: Includes source information, confidence scores, and quality metrics

The dataset can be found [here](https://huggingface.co/datasets/yudhiesh/cerberus-guardrails-small) on Huggingface.

<div style="width: 100%; overflow-x: auto; border: 1px solid #ddd; border-radius: 4px;">
<iframe
  src="https://huggingface.co/datasets/yudhiesh/cerberus-guardrails-small/embed/viewer/default/train"
  title="Cerberus training dataset"
  loading="lazy"
  frameborder="0"
  width="100%"
  height="560px"
></iframe>
</div>

## Exploratory Data Analysis on Synthetic Data

To better understand the characteristics of the synthetic dataset, a thorough analysis was conducted using [BERTopic](https://maartengr.github.io/BERTopic/), a powerful topic modeling tool. An interactive analysis notebook was developed using [Marimo](https://marimo.io) (available in the [EDA directory](https://github.com/yudhiesh/cerberus/tree/master/eda)) to explore the patterns and distributions within the 566 examples.

<div align="center">
  <figure>
    <video width="800" autoplay loop muted playsinline>
      <source src="https://user-images.githubusercontent.com/25746895/218420473-4b2bb539-9dbe-407a-9674-a8317c7fb3bf.mp4" type="video/mp4">
    </video>
    <figcaption><em>BERTopic in action: Visualizing the dynamic clustering process of the dataset's topics</em></figcaption>
  </figure>
</div>

> The interactive views are optional. You can [open the dataset](https://huggingface.co/datasets/yudhiesh/cerberus-guardrails-small) or [open the analysis]({{ "/assets/eda.html" | absolute_url }}) directly, including from a feed reader.

The analysis revealed several interesting insights about the dataset's structure:

1. **Topic Distribution**: Using `thenlper/gte-large` embeddings and HDBSCAN clustering, distinct clusters were identified that naturally separated into safe and unsafe categories. This clustering validates the synthetic data generation approach, showing clear semantic boundaries between benign and potentially harmful content.

2. **Safe Content Clusters**: The safe examples formed coherent topic groups around legitimate technical domains:
   - Streaming systems (Kafka, real-time processing)
   - Database operations (Redis, caching strategies)
   - Application development (gaming, web services)
   These clusters demonstrate good coverage across various technical domains while maintaining clear safety boundaries.

3. **Unsafe Content Patterns**: The analysis also revealed well-defined clusters of potentially harmful content:
   - Network security exploits
   - Database injection attempts
   - System compromise techniques
   This clustering helps validate the unsafe example generation and ensures comprehensive coverage of potential threats.

The interactive visualization below shows the topic distribution and similarity relationships across the dataset. The clear separation between safe and unsafe clusters suggests the synthetic data generation process successfully created distinct, well-labeled examples suitable for training safety classifiers.

<div style="width: 100%; position: relative;">
  <div style="position: absolute; top: 10px; right: 10px; z-index: 10;">
    <a href="{{ '/assets/eda.html' | relative_url }}" 
       target="_blank" 
       class="embed-link">
      Open analysis ↗
    </a>
  </div>
  <div style="width: 100%; overflow-x: auto; border: 1px solid #ddd; border-radius: 4px;">
    <iframe src="{{ '/assets/eda.html' | relative_url }}" 
            title="Interactive synthetic data analysis"
            loading="lazy"
            width="100%"
            height="720px"
            frameborder="0">
    </iframe>
  </div>
</div>

# Final Thoughts

Getting examples from an LLM was the straightforward part. Deciding which examples to keep took more care: 1,000 generated prompts became 708 after deduplication, and disagreements between the generator and evaluator still needed a closer look.

The generation and evaluation runs cost around $20–25. Distilabel's caching helped while I was iterating, and Marimo made it easier to inspect the data alongside the topic analysis. Those tools were useful, but none of them removed the need to look at the examples themselves.

Before scaling this further, I would work on better coverage of rare cases and clearer checks for label quality. I would also compare the downstream classifier on held-out examples. A tidy dataset is useful; whether it helps the classifier is the next question.

# Coming Up Next

In Part 2 of this series, the synthetic dataset will be put to the test by comparing various algorithms for prompt safety classification. The evaluation will explore different approaches—from lightweight models to sophisticated neural architectures—analyzing their performance, speed, and resource requirements to find the optimal balance for real-world deployment. Stay tuned!

# References
- [Guardrails AI](https://www.guardrailsai.com/docs)
- [How to use Guardrails from OpenAI](https://cookbook.openai.com/examples/how_to_use_guardrails)
- [What is Synthetic Data?](https://aws.amazon.com/what-is/synthetic-data/)
- [Synthetic Data Generation (Part 1)](https://cookbook.openai.com/examples/sdg1)
- [Distilabel](https://distilabel.argilla.io/latest/)
- [SemHash](https://github.com/MinishLab/semhash)
- [DeepEval](https://deepeval.com/)
- [Argilla](https://argilla.io/)
- [BERTopic](https://maartengr.github.io/BERTopic/)
- [Marimo](https://marimo.io)
- [G-Eval Guide](https://www.confident-ai.com/blog/g-eval-the-definitive-guide)
- [Cerberus Dataset](https://huggingface.co/datasets/yudhiesh/cerberus-guardrails-small)
- [Constitutional Classifiers: Defending against Universal Jailbreaks](https://arxiv.org/abs/2501.18837)
