---
layout: post
title: "The Achilles Heel of Vector Search: Filters"
date: 2025-05-09 12:00:00 +0000
tags: ml
mathjax: true
---
# Table of Contents
- [Overview](#overview)
- [References](#references)

# Overview

Back in Q2 2024 at my previous company, I set out to answer what should have been a simple question for the whole company:  
> “Which vector database should we use?”

What started as a quick investigation turned into a deep dive—presenting my findings in talks (slides[^7],[^8]) and questioning everything I thought I knew about indexes and search. Along the way, one discovery really blew me away: **unlike a traditional RDBMS, adding a filter to a vector search often *slows* it down**, not speeds it up. In an SQL database, you’d apply a B-tree filter and instantly prune away 90% of the rows—faster queries and higher throughput. But in a vector index, that same filter can break the ANN structure’s assumptions and force you into brute-force work or awkward workarounds.

In this post, we’ll unpack why **filtered vector search** is the Achilles’ heel of ANN:  
- **Pre-filtering:** why “filter then search” often collapses back into brute force  
- **Post-filtering:** how “search then filter” can miss results or blow up latency  
- **Integrated filtering:** the cutting-edge tweaks (Qdrant’s filterable HNSW, Weaviate’s ACORN, Pinecone’s single-stage) that restore both speed and accuracy  

We’ll also look at benchmark numbers from Faiss, Qdrant, Pinecone and Weaviate—comparing throughput and P95 latency with and without filters—and finish by exploring a promising new idea: **filter fusion**, where you encode metadata into your embeddings so that standard ANN search magically respects your filter without any extra step.

---

**Note:**
- **X-axis:** mean_precision  
- **Ratio 0.1:** filter returns 10% of the original dataset, highly selective filter  
- **Throughput chart:** Y-axis is **Throughput (QPS)**  
- **Latency chart:** Y-axis is **P95 Latency (seconds)**  

---

**Throughput Across Different Vector Databases**  
![Throughput Across Different Vector Databases between filtered and non-filtered search](https://media.licdn.com/dms/image/v2/D5622AQGRCJJV2y54sQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1723180122713?e=1749686400&v=beta&t=uDrM7_Ipu0V5opGTwrnvlaY-5rt3N8oYsWewpWBSdfo)

**QPS Across Different Vector Databases**  
![QPS Across Different Vector Databases between filtered and non-filtered search](https://media.licdn.com/dms/image/v2/D5622AQG_PWx5sHzp8g/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1723180122814?e=1749686400&v=beta&t=MYrgrUOS6ohyy64zJJCQD9jhbq95LgSwPB_0lHi3Z2A)

After running these benchmarks across six vector engines, two query modes (regular vs. 10%-filtered), and up to 16 search threads, a few clear patterns emerged:

- **Thread scaling boosts throughput for all engines**, but especially for Pinecone-p2 and Zilliz-AUTOINDEX. At 16 threads, Pinecone tops out near ~800 QPS unfiltered and ~600 QPS filtered, while Zilliz reaches ~750 QPS unfiltered and ~700 QPS filtered.  
- **Filtering often *improves* throughput** for systems with integrated filtering:  
  - Pinecone-p2 and Zilliz-AUTOINDEX both show a ~1.2×–1.5× throughput bump under the 10% filter, since they prune the search space in-flight.  
  - Qdrant-HNSW sees a modest gain as well, thanks to its augmented-graph strategy.  
- **Engines that fall back to brute-force filtering** (e.g. LanceDB-IVF_PQ-ratio-0.1) or pure post-filtering (OpenSearch-HNSW-ratio-0.1, PGVector-HNSW-ratio-0.1) gain little or even **lose** throughput when a filter is applied, because they scan or oversample the subset.  
- **Tail latencies (P95) remain sub-100 ms** for most engines in regular search—even at high precision—and often **drop further under a 10% filter** for integrated-filter systems.  
  - Pinecone-p2 and Zilliz-AUTOINDEX reduce P95 by 10–30 ms when filtering, keeping latencies around 30–50 ms at 16 threads.  
  - Qdrant-HNSW-ratio-0.1 holds roughly the same latency as its unfiltered case, showing its planner chooses the optimal strategy.  
  - By contrast, LanceDB-IVF_PQ-ratio-0.1 and OpenSearch-HNSW-ratio-0.1 can spike up to 200–300 ms under filtering, indicating a brute-force or oversampling penalty.  

# Filtering Strategies in ANN Search

In vector search, a filter means we only want results from a subset of vectors (e.g. only vectors whose associated category is “laptop”). Formally, *vector filtering consists in returning only database vectors that meet some criterion; other vectors are ignored.*

There are three broad strategies to combine filtering with ANN (Approximate Nearest Neighbor) search:

- **Pre-filtering (filter-then-search)**[^1][^2]: First narrow down the dataset to vectors matching the metadata filter, then perform the vector similarity search on that subset. This guarantees results satisfy the filter, but if the subset is large, the vector search becomes slower (potentially a brute-force scan).

- **Post-filtering (search-then-filter)**[^3][^4]: First run the vector search on the full dataset (or a broad portion), then filter out any results that don’t meet the criteria. This leverages the ANN index fully, but you might need to retrieve many extra candidates to end up with enough filtered results. If the filter is very selective, post-filtering can miss results or require heavy oversampling, hurting accuracy and latency.

- **In-algorithm filtering (integrated or single-stage filtering)**[^5][^6]: Incorporate the filter into the ANN search process itself, so that the algorithm only (or mostly) explores vectors that satisfy the filter. This can be done by modifying the index or search logic to be “filter-aware” (e.g. marking or connecting vectors by filter). The goal is to get the accuracy of pre-filtering without its speed penalty, by effectively performing filtering and ANN search in one stage.

Each approach involves trade-offs, which vary by index type. Let’s see how graph-based indices (HNSW) and inverted file indices (IVF/IVF-PQ) handle filtering under these strategies.

# References

[^1]: Pinecone. “[Vector Search Filtering Strategies](https://www.pinecone.io/learn/vector-search-filtering)”  
[^2]: Qdrant. “[Vector Search Filtering](https://qdrant.tech/articles/vector-search-filtering)”  
[^3]: Pinecone. “[The Pitfalls of Post-Filtering in ANN](https://www.pinecone.io/learn/vector-search-filtering#post-filtering)”  
[^4]: Qdrant. “[Why Post-Filtering Wastes Computation](https://qdrant.tech/articles/vector-search-filtering#post-filtering)”  
[^5]: Qdrant. “[Filterable HNSW: In-Algorithm Filtering](https://qdrant.tech/articles/vector-search-filtering#filterable-hnsw)”  
[^6]: Weaviate. “[Speed Up Filtered Vector Search (ACORN)](https://weaviate.io/blog/speed-up-filtered-vector-search)”  
[^7]: Ravindranath, Yudhiesh. “[Vector Databases Talk (LinkedIn)](https://www.linkedin.com/posts/yudhiesh-ravindranath_vectordatabases-machinelearning-datascience-activity-7227541287279718400-VzIA)”  
[^8]: Ravindranath, Yudhiesh. “[Grokking Vector Databases (Slides)](https://drive.google.com/file/d/1Hgfhf1iT-I3G4Q3j2siKj0GSarDFOJD-/view?usp=sharing)”

