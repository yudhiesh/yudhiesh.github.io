---
layout: post
title:  "Feature Flag Driven Development in Machine Learning"
date:   2024-12-22 14:30:28 +0800
categories: ML SWE
---
# Table of Contents
- [Overview](#overview)
- [How Search Works](#how-search-works)
- [Semantic Search vs Keyword Search](#semantic-search-vs-keyword-search)
- [How Semantic Search Works](#how-semantic-search-works)
 - [Word Embeddings](#word-embeddings)
 - [Vector Databases](#vector-databases)
- [Semantic Search Implementation](#how-semantic-search-works-1)
 - [Offline Process](#offline)
   - [Document Embedding](#step-1)
 - [Online Process](#online)
   - [Query Embedding](#step-1-1)
   - [Similarity Search](#step-2)
- [References](#references)

# Overview
In my prior role I was working on building out our companies search engine where users could search across our entire product suite and hopefully showcase answers/products that are relevant to the users search query. This system was interesting in the sense that it combined regular search that you would experience with using Google Search + GenAI, much like how Google Search works right now:
![User asking SGE to evaluate two national parks that are best for young kids and a dog](https://storage.googleapis.com/gweb-uniblog-publish-prod/images/IOM_BryceCanyon_Desktop_Launch.width-1000.format-webp.webp)

Breaking this down we can split the main important parts of building out the entire search experience for users into:
1. Query - The user's query 
2. GenAI Search Results - Output from the GenAI Search Service results, powered by an LLM
3. "Regular" Search Results - Output from the Regular Search Results

![Image]({{ site.baseurl }}/images/Screenshot 2024-12-22 at 15.46.05.png)

In terms of a graph, we can see that the `Query` is sent to the `Frontend` which then sends concurrent requests to both Search services:
![Image]({{ site.baseurl }}/images/Screenshot 2024-12-22 at 15.48.08.png)

I won't be diving into the GenAI Search Service details in this post. Instead, I'll focus how I tried to improve our "Regular" Search Service while navigating several key constraints:

1. *Time* - Like most startups, features needed to ship "yesterday"
2. *System Reliability* - The search revamp was mission-critical. Poor results could drive users away from adopting the new system entirely.
3. *Complex Integration Requirements* - Beyond standard search complexity, we needed to integrate machine learning models to improve result relevance.

# How Search Works
The next section is a condensed version of this [amazing blog post from Eugene Yan](https://eugeneyan.com/writing/system-design-for-discovery/). 

![2x2 of online vs. offline environments, and candidate retrieval vs. ranking.](https://eugeneyan.com/assets/discovery-2x2.jpg "2x2 of online vs. offline environments, and candidate retrieval vs. ranking.")
Search Engines are very complex pieces of software which can be condensed down into the following two stages:
1. *Retrieval* - Fetching a representative sample of the items to search through. For example, social media platforms such as TikTok and Instagram would pull shot-form video's. I say a representative sample as, in most cases in Search and Recommendations its impossible to perform a greedy search across all possible items when the # of items to search through is in the billions/trillions. Retrieval is focused on being fast and doing a good enough job considering the fact that the items to search through is massive, i.e., 1M -> 100 items. You could just show the results here but they wouldn't be that relevant as we would like due to the fact that we are not encoding any additional features that could improve search results.
2. *Ranking* - Of the representative sample that was returned lets turn them into a better set of ranked items incorporating user/item/contextual features that could help give them a more optimal ordering. This step is much slower than the retrieval stage.
In each of the two stages we need to perform actions in both offline and online settings:
1. *Offline* - Ahead of time before the results are even shown. For example, we could be ingesting documents to search over into our search database.
2. *Online* - When a user's query is to be answered we will need to perform these steps in order to generate a response. For example, we would need to retrieve the items we want and then pass them over to the ranking stage which reranks them and then finally return the results.

One thing to note about the prior diagram is that this is a very Machine Learning centric search system, where we utilise Machine Learning models for the ranking stage or using embedding models to convert natural language text into numerical representations that a computer can understand while encoding intricacies related to languages. This kind of systems are setup after much iterations on basic keyword search such as inverted indexes, tf-idf, bm25 and so on. The main points in the 2x2 matrix still apply here.

Since we were in the early stages of improving search relevance the initial implementation used OpenSearch running on the AWS's Managed OpenSearch offering with [BM25 ranking](https://en.wikipedia.org/wiki/Okapi_BM25), which worked reasonably well for exact matches. 
![Image]({{ site.baseurl }}/images/CH03_F07_Grainger.png)

We supplemented the use of it with custom re-rankers that would apply heuristics in order to further re-rank items incorporating business requirements where we could add in our own sort of boosting logic where the business sees fit. For example, in the context of an e-commerce store you might want to add in custom boosting logic:
```json
{
 "query": {
   "function_score": {
     "query": {
       "multi_match": {
         "query": "running shoes",
         "fields": ["title", "description", "brand"]
       }
     },
     "functions": [
       {
         "exp": {
           "created_date": {
             "origin": "now",
             "scale": "30d",
             "decay": 0.5
           }
         },
         "weight": 2
       },
       {
         "filter": {
           "range": {"inventory_count": {"gt": 5}}
         },
         "weight": 1.5
       },
       {
         "field_value_factor": {
           "field": "rating",
           "factor": 1.2
         }
       }
     ],
     "score_mode": "sum",
     "boost_mode": "multiply"
   }
 }
}
```
The prior query would:
- boost new arrivals
- prioritize in-stock items
- factor in product rating
- etc.

In the Machine Learning space, you would learn these sorts of interactions instead via training a Learning to Rank(LTR) model using features related to click-through rates, conversion rates, etc. 

## Semantic Search vs Keyword Search
The issue with Keyword Search and BM25(builds off TF-IDF) is that user queries aren't as black and white as finding the most relevant document looking into:
- how frequently a term appears in a document(TF)
- how unique that term is across all documents(IDF)

Keyword search fails to understand the context and meaning behind queries. A search for "affordable running shoes" might miss relevant results for "budget athletic footwear" despite their semantic similarity. This is where embeddings come in - they represent words and phrases as dense vectors in high-dimensional space, capturing semantic relationships and contextual nuances. By encoding meaning rather than just matching exact terms, embeddings enable search systems to understand that "cozy apartment" and "comfortable flat" are conceptually similar, even without shared keywords.
![Lexical search vs. Semantic search comparison](https://static.semrush.com/blog/uploads/media/31/50/3150dd9c369ec2c272658bdfb161ad3f/d78b61c1a1a21dac3d16147e9cb4852e/FLhzdoFxIHH23S-htv5mDsTi-bzzpric_UPiNDG8EH8TZvqEqv3FaXNVA4dNjMzXTK09stsR7mGjTH4TfJAEfPZN_KE91ZUND-6swWj9VFhtdMPNAyyFHq9sSdvxiBvHzhNFnExJBVVL5ZXupob8Cpc.png)

## How Semantic Search Works
Semantic Search requires the following:
- Word Embeddings
- Vector Database

### Word Embeddings
Embeddings which are the core building block of any NLP model was first highlighted via the seminal paper [Word2Vec](https://arxiv.org/abs/1301.3781). The key breakthrough was showing that words with similar meanings would cluster together in vector space, enabling mathematical operations on language (like "king - man + woman = queen"). This distributional hypothesis - that words appearing in similar contexts have similar meanings - provided a foundation for modern NLP.
![https://medium.com/@vipra_singh/llm-architectures-explained-word-embeddings-part-2-ff6b9cf1d82d](https://miro.medium.com/v2/resize:fit:1400/1*2YXem_lD24XW8VU11CCVZA.jpeg)

### Vector Databases
A vector database indexes and stores vector embeddings for fast retrieval and similarity search, with capabilities like CRUD operations, metadata filtering, horizontal scaling, and serverless.
![https://www.pinecone.io/learn/vector-database/](https://www.pinecone.io/_next/image/?url=https%3A%2F%2Fcdn.sanity.io%2Fimages%2Fvr8gru94%2Fproduction%2Fe88ebbacb848b09e477d11eedf4209d10ea4ac0a-1399x537.png&w=1920&q=75)
There are a ton of vector database options out there, largely split into purpose built vector databases and databases with vector extensions added to them.
![](https://media.licdn.com/dms/image/D5612AQFPNZxykG6Enw/article-cover_image-shrink_600_2000/0/1688152025187?e=2147483647&v=beta&t=_iGesp-lS22luisdy9BqvIXaYlTx8EfIQNgL4taflQY)
Picking a vector database to use warrants its own blog post due to the sheer number of options and considerations that need to take place.

## How Semantic Search Works?
Going back to Eugene's terrific blog post we can see how a Semantic Search fits into the big picture by looking at the Retrieval section covering the offline and online side of search. 
![Image]({{ site.baseurl }}/images/Screenshot 2024-12-22 at 18.33.38.png)

### Offline

#### Step 1
A document comes in and is transformed from its format such as text/image into an embedding using an embedding model such as [thenlper/gte-large](https://huggingface.co/thenlper/gte-large). For example, lets say that we are trying perform semantic search over the `review_body` in the following document:
```json
{
  "review_id": "en_0802237",
  "product_id": "product_en_0417539",
  "reviewer_id": "reviewer_en_0649304",
  "stars": "3",
  "review_body": "I love this product so much i bought it twice! But their customer service is TERRIBLE. I received the second glassware broken and did not receive a response for one week and STILL have not heard from anyone to receive my refund. I received it on time, but am not happy at the moment.",
  "review_title": "Would recommend this product, but not the seller if something goes wrong.",
  "language": "en",
  "product_category": "kitchen"
}
```
After we run our embedding model on the field that we want to embed, we would get a new field in the document, `review_body_embedding` which will contain a vector of a fixed dimension with floats where the fixed dimension is dependent on the embedding model we use `1024`. 
*NOTE:* Embeddings will be truncated in the examples.
```json
{
  "review_id": "en_0802237",
  "product_id": "product_en_0417539",
  "reviewer_id": "reviewer_en_0649304",
  "stars": "3",
  "review_body": "I love this product so much i bought it twice! But their customer service is TERRIBLE. I received the second glassware broken and did not receive a response for one week and STILL have not heard from anyone to receive my refund. I received it on time, but am not happy at the moment.",
  "review_body_embedding": [0.1234, -0.1122, 0.8889],
  "review_title": "Would recommend this product, but not the seller if something goes wrong.",
  "language": "en",
  "product_category": "kitchen"
}
```
### Online

#### Step 1

When a user's query comes in we will then convert that into an embedding using the exact same model that was used to embed the documents, this is a very important step. 
```json
{
  "query": "Find me reviews related to glassware"
}
```
After embedding the user's query you will get the following input document:
```json
{
  "query": "Find me reviews related to glassware",
  "query_embedding": [0.2212, 0.9221, -0.1111]
}
```
#### Step 2

# References
