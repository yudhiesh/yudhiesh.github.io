---
layout: post
title: "The Achilles Heel of Vector Search: Filters"
date: 2025-05-09 12:00:00 +0000
tags: ml
mathjax: true
---
# Table of Contents

* [Overview](#overview)
* [Filtering Strategies in ANN Search](#filtering-strategies-in-ann-search)
* [Filtering in HNSW (Graph-Based Indexes)](#filtering-in-hnsw-graph-based-indexes)
* [Filtering in IVF and IVF-PQ (Inverted Indexes)](#filtering-in-ivf-and-ivf-pq-inverted-indexes)
* [How Vector Databases Support Filtering](#how-vector-databases-support-filtering)
* [Filtered vs Unfiltered Search Performance](#filtered-vs-unfiltered-search-performance)
* [Why Filtered Vector Search Is Slower than Traditional Filtering](#why-filtered-vector-search-is-slower-than-traditional-filtering)
* [Filter Fusion: Encoding Filters into Embeddings](#filter-fusion-encoding-filters-into-embeddings)
* [References](#references)


# Overview

Back in Q2 2024 at my previous company, I set out to answer what should have been a simple question for the whole company:  
> “Which vector database should we use?”

What started as a quick investigation turned into a deep dive—presenting my findings in talks (slides[^7],[^8]) and questioning everything I thought I knew about indexes and search. Along the way, one discovery really blew me away: **unlike a traditional RDBMS, adding a filter to a vector search often *slows* it down**, not speeds it up.

In this post, we’ll unpack why **filtered vector search** is the Achilles’ heel of ANN:  
- **Pre-filtering:** why “filter then search” often collapses back into brute force  
- **Post-filtering:** how “search then filter” can miss results or blow up latency  
- **Integrated filtering:** the cutting-edge tweaks (Qdrant’s filterable HNSW, Weaviate’s ACORN, Pinecone’s single-stage) that restore both speed and accuracy  

We’ll also look at benchmark numbers from Faiss, Qdrant, Pinecone and Weaviate—and finish by exploring a promising new idea: **filter fusion**, where you encode metadata into your embeddings so that standard ANN search magically respects your filter without any extra step.

---

**Note:**  
- **X-axis:** `mean_precision`  
- **Ratio 0.1:** filter returns 10% of the original dataset (highly selective)  
- **Throughput chart:** Y-axis is **Throughput (QPS)**  
- **Latency chart:** Y-axis is **P95 Latency (seconds)**  

---

## Throughput Across Different Vector Databases  
![Throughput Across Different Vector Databases between filtered and non-filtered search](https://media.licdn.com/dms/image/v2/D5622AQGRCJJV2y54sQ/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1723180122713?e=1749686400&v=beta&t=uDrM7_Ipu0V5opGTwrnvlaY-5rt3N8oYsWewpWBSdfo)

## QPS Across Different Vector Databases  
![QPS Across Different Vector Databases between filtered and non-filtered search](https://media.licdn.com/dms/image/v2/D5622AQG_PWx5sHzp8g/feedshare-shrink_2048_1536/feedshare-shrink_2048_1536/0/1723180122814?e=1749686400&v=beta&t=MYrgrUOS6ohyy64zJJCQD9jhbq95LgSwPB_0lHi3Z2A)

After running these benchmarks across six vector engines, two query modes (regular vs. 10%-filtered), and up to 16 search threads, a few clear patterns emerged:

- **Thread scaling boosts throughput for all engines**, but especially for Pinecone-p2 and Zilliz-AUTOINDEX. At 16 threads, Pinecone hits ~800 QPS unfiltered and ~600 QPS filtered; Zilliz reaches ~750 QPS unfiltered and ~700 QPS filtered.  
- **Filtering often *improves* throughput** on systems with integrated filtering:  
  - Pinecone-p2 and Zilliz-AUTOINDEX each see a ~1.2×–1.5× throughput bump under the 10% filter.  
  - Qdrant-HNSW shows a smaller gain thanks to its augmented-graph strategy.  
- **Brute-force or post-filter engines lose ground** under a tight filter: LanceDB-IVF_PQ-ratio-0.1, OpenSearch-HNSW-ratio-0.1 and PGVector-HNSW-ratio-0.1 all dip in QPS when filtering, since they scan or oversample the subset.  
- **Tail latencies (P95) stay sub-100 ms** in regular searches and often **drop further** with a 10% filter on integrated systems:  
  - Pinecone-p2 and Zilliz-AUTOINDEX shave 10–30 ms off P95, landing around 30–50 ms at 16 threads.  
  - Qdrant-HNSW-ratio-0.1 maintains similar latency, thanks to its adaptive planner.  
  - Brute-force fallbacks (LanceDB-IVF_PQ-ratio-0.1, OpenSearch-HNSW-ratio-0.1) can spike to 200–300 ms under filtering.

**Bottom line**: Engines with **in-algorithm filtering** not only preserve recall—they actually get faster, more predictable searches when filters prune the workload. Engines relying on pre- or post-filter fallbacks may see little benefit or even performance degradation when filters are applied.

# Filtering Strategies in ANN Search

In vector search, a *filter* means we only want results from a subset of vectors (e.g. those whose category is “laptop”). Formally, vector filtering **returns** only vectors that meet a criterion and ignores all others. There are three broad strategies to combine filtering with ANN (Approximate Nearest Neighbor) search:;

* **Pre-filtering (filter-then-search)**
  First apply the metadata filter (e.g. via an inverted index), then run ANN on the resulting subset. Guarantees correctness, but if that subset is large you end up doing a near-linear scan (brute force).

* **Post-filtering (search-then-filter)**
  First run ANN on the full dataset, then drop any results that fail the filter. Fast when filters are loose, but restrictive filters require oversampling (searching many more neighbors) to avoid missing matches, which hurts latency and recall.

* **In-algorithm filtering (integrated or single-stage)**
  Modify the index or search logic so the ANN search itself only traverses or returns filtered vectors—e.g. Qdrant adds extra graph links, Weaviate’s ACORN does two-hop expansions, Pinecone merges metadata and vector indexes. This aims to combine pre-filter accuracy with ANN speed in a single pass.


# Filtering in HNSW (Graph-Based Indexes)

HNSW (Hierarchical Navigable Small World) is a popular graph index for ANN. Vectors are nodes connected by edges; search traverses this graph greedily to find nearest neighbors. Applying filters here poses unique challenges:;

### Pre-Filtering

Skip the graph entirely when a filter is present: retrieve the IDs matching the filter (via a secondary metadata index) and do a linear scan or a small ANN build over that subset. Fast if the filtered subset is tiny, but performance degrades linearly as selection grows.

### Post-Filtering

Run standard HNSW ignoring the filter, then discard non-matching neighbors. To ensure *k* valid results, you must oversample (e.g. request 10× *k* candidates if only 10 % match). This wastes distance computations and can miss results under tight filters.

### Integrated Filtering

Make the HNSW graph itself filter-aware. 

![Filterable vector index](https://qdrant.tech/articles_data/vector-search-filtering/filterable-vector-index.png)

Approaches include:
* **Sweeping (Weaviate):** traverse normally but check each candidate’s filter, extending the search until enough valid points are found.
* **Filterable HNSW (Qdrant):** add extra intra-category links so filtered nodes don’t break graph connectivity.
* **ACORN (Weaviate):** maintain unpruned graph edges and use two-hop jumps to skip filtered-out nodes without disconnecting the graph.

#### HNSW Filtering Comparison

| Approach       | Recall            | Latency                  | Memory Overhead           | Complexity                |
| -------------- | ----------------- | ------------------------ | ------------------------- | ------------------------- |
| Pre-filtering  | 100 % (subset)    | O(subset size)           | Low (metadata index)      | Low                       |
| Post-filtering | Variable (<100 %) | O(ANN + oversample)      | None                      | Low (tuning needed)       |
| Integrated     | ≈ 100 %           | Near unfiltered (pruned) | High (extra links, edges) | High (algorithmic change) |


# Filtering in IVF and IVF-PQ (Inverted Indexes)

IVF (Inverted File) clusters data into coarse buckets; IVF-PQ adds vector compression. Filtering here mirrors HNSW strategies but at the cluster level:;

### Pre-Filtering

Partition the IVF index by filter value (one index per category) or retrieve IDs via a metadata index and scan only those buckets. Efficient for tiny subsets; impractical at scale or high-cardinality.

### Post-Filtering

Perform standard IVF search (probe *m* nearest centroids), then drop non-matching items. Requires increasing `nprobe` under strict filters, which raises latency.

### Integrated Filtering

Tag centroids with filter values and skip entire clusters that contain no valid vectors. This single-stage approach prunes the search space upfront, combining accuracy with speed.

#### IVF / IVF-PQ Filtering Comparison

| Approach       | Recall   | Latency                                 | Memory/Index Overhead   | Complexity             |
| -------------- | -------- | --------------------------------------- | ----------------------- | ---------------------- |
| Pre-filtering  | 100 %    | O(subset size)                          | Metadata postings lists | Low                    |
| Post-filtering | Variable | O(IVF + oversample)                     | None                    | Low (parameter tuning) |
| Integrated     | ≈ 100 %  | Reduced by skipping irrelevant clusters | Cluster tag bitmasks    | Medium to High         |


# How Vector Databases Support Filtering

* **Faiss:** Offers ID-mask support (bitsets) to skip candidates by metadata.
* **Pinecone:** Merges vector and metadata indexes into a single-stage filterable system.
* **Qdrant:** Uses a payload (inverted) index plus filterable HNSW, with a planner that picks brute-force or graph filtering based on selectivity.
* **Weaviate:** Combines HNSW + ACORN, falling back to flat scans on tiny subsets and two-hop graph jumps otherwise.;


# Filtered vs Unfiltered Search Performance

**Faiss (IVF) – NeurIPS’23 Filtered Search Track:**
A Faiss baseline on 10 M vectors hit \~3 k QPS at 90 % recall. Optimized methods (e.g. multi-level IVF-PQ, graph hybrids) reached 30–40 k QPS at the same recall—a 10× gap.;

**Weaviate (HNSW) – Pre-ACORN vs ACORN:**
Pre-ACORN used flat scans for small filters and HNSW with inline sweeping for larger ones, leading to unpredictable latency under correlated filters. ACORN’s dense graphs and two-hop expansions stabilized performance and preserved recall on highly selective queries.;

**Pinecone – Single-Stage Filtering:**
On 1.2 M vectors (768-dim), unfiltered queries averaged 79 ms; a 50 % selective filter dropped latency to 57 ms, and a 1 % filter to 51.6 ms (\~35 % faster) by truly reducing search work.;

**Qdrant – Filterable HNSW:**
Qdrant’s planner picks graph filtering vs brute-force based on “filter cardinality.” Users report negligible recall loss and minimal query-time increase for moderately selective filters, avoiding worst-case slowdowns.;


# Why Filtered Vector Search Is Slower than Traditional Filtering
Given the performance challenges above, a fair question is: why is this so hard? **Relational databases** have provided fast filtering for decades using structures like B-trees, hash indexes, etc. If you ask SQL for `SELECT * FROM products WHERE category = 'laptop'` , it can use an index on the category column to fetch matching rows in almost constant time relative to the result size. So why can’t vector searches do `top_k ANN ... WHERE category='laptop'` just as easily?
The core issue is that vector search and metadata filtering are fundamentally different operations that are not trivially compatible. A few reasons:

1. **Index Mismatch:** ANN indexes optimize for proximity, not boolean predicates—no native “category” tree.
2. **Graph Disconnects:** Removing nodes fragments HNSW; B-trees never face this.
3. **No Composite Key:** You can’t jointly index (vector, metadata) as easily as relational DBs can on two columns.
4. **Approximation vs Exactness:** ANN trades recall for speed; filters demand exact matches, forcing extra work.
5. **Join-Like Complexity:** Filtering + similarity is akin to a join between vector and metadata indexes, which is inherently more complex.;


# Filter Fusion: Encoding Filters into Embeddings

Given the difficulties above, an intriguing solution is **Filter Fusion** – essentially, bake the filter criteria into the vector representation itself. The idea is to avoid explicit filtering altogether by designing embeddings (or distance functions) such that an item that doesn’t meet the filter would never appear in the top results in the first place.

Imagine if for each vector, we could append a small descriptor of its metadata, and for each query, do the same. During similarity search, only vectors whose descriptors match the query’s descriptor closely would score highly. This way, the vector similarity metric itself enforces the filter. It’s like secretly encoding `“category = laptop”` within the vector coordinates.

**Motivation & Intuition**: If the vector space can incorporate both content and metadata, then a single nearest neighbor search can handle both aspects. This eliminates the need for separate filtering logic and ensures zero overhead for applying filters – the math of similarity takes care of it. It also potentially means we can use off-the-shelf ANN methods without modification, because we’re just searching in a slightly higher-dimensional space (original embedding + metadata encoding).

One simple approach to filter fusion is vector concatenation. For example, say each data vector is 256- dim and we have a categorical filter with 10 possible values. We can allocate an extra 10 dimensions (a one-hot encoding for the category). A vector for an item in category 3 would have those extra 10 dims all 0 except a 1 in position 3. A query that requires category 3 would similarly have that one-hot in its vector. If we use standard Euclidean or cosine distance on the combined (266-dim) vectors, a candidate from a different category will have a large distance on the metadata dimensions, likely pushing it out of the top results. We can even exaggerate this by weighting the metadata dimensions more strongly (e.g. set the one-hot to some large value) so that any category mismatch dominates the distance. In effect, the nearest neighbor search “prefers” vectors of the same category because others are far away in those extra dimensions.

**Filter fusion in practice**: Let’s illustrate a basic example in Python pseudocode. We’ll create some 2D points with an associated filter value (say, color being Red or Blue), embed the filter into the vector, and perform a fused search:
```python
import numpy as np

# 2D vectors + binary filter (0=Red, 1=Blue)
vectors = np.random.rand(1000, 2)
filters = np.random.randint(0, 2, size=(1000, 1)) * 10.0  # high weight
aug_vectors = np.hstack([vectors, filters])

# Query requiring filter = 1 (Blue)
query = np.array([0.5, 0.5])
query_filter = np.array([1.0]) * 10.0
aug_query = np.hstack([query, query_filter])

# Euclidean NN on augmented vectors naturally prefers same-filter points
dists = np.linalg.norm(aug_vectors - aug_query, axis=1)
nearest = np.argmin(dists)
```

By concatenating metadata as weighted dimensions, off-the-shelf ANN can perform filtered search in one pass, eliminating extra filter logic.

# References

