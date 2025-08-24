# RCA-Net: A Context-Aware Relational Network for Sentiment Analysis in the Metaverse.

## Introduction

This repository accompanies our work on context-aware sentiment analysis and text classification in metaverse-like textual settings. 
We combine scenario-driven relational modeling with topic–emotion alignment to improve performance and reproducibility across multiple datasets.
Scenario-driven graph reasoning + multi-layer embeddings for sentiment–topic alignment on heterogeneous text (movie reviews, SMS, Twitter, literature).
~4% gain over baselines, 0.93 on 3D text-generation task.

## Project Overview

- Goal: Robust sentiment–topic analysis and text classification under heterogeneous, non-uniform text.

- RCA-Net fuses scenario-aware graph reasoning with CNN/RNN heads and a consistency regularizer across topic–emotion layers.


## Algorithms and Techniques

- **Bayesian Components**: Probabilistic modeling of topic–emotion posteriors (via LDA + Gibbs sampling).
- **Deep Models**: Joint multi-view representation learning by combining CNN (topic features), RNN/LSTM (temporal-emotional flow), and GNN (relational-context graphs).

### Key Approaches

1. **Scenario-aware attention (Graph aggregation)**  
   Adaptive neighborhood weighting allows RCA-Net to model heterogeneous dependencies:  

 $$
h_v^{(t+1)} = \sigma\left(W h_v^{(t)} + \sum_{u \in \mathcal{N}(v)} \alpha_{uv} h_u^{(t)}\right),
\quad
\alpha_{uv} = \mathrm{softmax}_u\left(\mathrm{LeakyReLU}\left(a^\top[W h_u \,\|\, W h_v]\right)\right)
$$


2. **Scenario-weighted pooling**  
   Scenario-specific node representations are aggregated with learned weights:  

$$
H_{\mathrm{GNN}}^{(s)} = \sum_{k=1}^K w_k^{(s)} \mathbf{h}_k^{(s)}
$$


3. **Consistency-regularized update**  
   A regularization term enforces alignment between sentiment–topic predictions:  

$$
h_v^{(t+1)} = \sigma \left( W_s h_v^{(t)} + \sum_{u \in \mathcal{N}(v)} \alpha_{uv} h_u^{(t)} \right)
+ \lambda \, \mathcal{L}_{\text{consistency}}
$$


4. **Task-specific heads**  
Sentiment head:
$$
\hat{y}_{\text{sent}} = \text{softmax} \big( \text{CNN}(H_{\text{GNN}}^{(s)}) \big)
$$


Topic head:
$$
\hat{y}_{\text{topic}} = \text{softmax} \big( \text{MLP}(H_{\text{GNN}}^{(s)}) \big)
$$




## Topic Modeling

- Number of topics `t ∈ {10, 20, 30}`  
- Dirichlet hyperparameters **α, β** tuned on dev set  
- Document–topic distribution `θ_d`, topic–word distribution `φ` estimated  

---


## Splits

- Default **80/20** split, stratified for class balance  
- **Twitter:** prefer time-aware split to avoid data leakage  
- Reproducibility: fix seeds (`42`) across numpy/torch  

---

## Embeddings & Optimization

- **Initialization:** Xavier/Glorot  
- **Optimizer:** `Adam(lr=1e-3, betas=(0.9, 0.999), weight_decay=1e-5)`  
- **Consistency weight λ:** search in `[0.001, 0.005]` (dev-set selection)  

---

## Hyperparameters

| Name              | Symbol | Value/Range     |
|-------------------|--------|-----------------|
| Max terms         | `mt`   | `9000`          |
| Decompositions    | `r`    | `{1, 2}`        |
| Topic groups      | `t`    | `{10, 20, 30}`  |
| Density adjustment| `k`    | `0.01`          |



### RCA-NET Pseudocode
- [RCA-Net_Pseduo_1.pdf](https://github.com/user-attachments/files/21957890/RCA-Net_Pseduo_1.pdf)
- [RCA-Net_Pseduo_2.pdf](https://github.com/user-attachments/files/21957891/RCA-Net_Pseduo_2.pdf)



## Results
### Review Analysis (Table 2)

| Model      | Topic Sim. | Positive | Negative |
|------------|------------|----------|----------|
| RCA-Net 1  | **0.75**   | **0.79** | **0.74** |
| RCA-Net 2  | **0.78**   | **0.81** | **0.76** |
| RCA-Net 3  | **0.80**   | **0.83** | **0.77** |
| LSTM       | 0.65       | 0.70     | 0.63     |
| CNN        | 0.69       | 0.72     | 0.65     |
| GNN-CNN    | 0.72       | 0.76     | 0.70     |
| GNN        | 0.70       | 0.74     | 0.71     |
| GRU        | 0.68       | 0.71     | 0.66     |

### Message Analysis (Table 3)

| Model      | Topic Sim. | Positive | Negative |
|------------|------------|----------|----------|
| RCA-Net 1  | **0.70**   | **0.75** | **0.72** |
| RCA-Net 2  | **0.73**   | **0.77** | **0.74** |
| RCA-Net 3  | **0.75**   | **0.79** | **0.76** |
| LSTM       | 0.60       | 0.65     | 0.62     |
| CNN        | 0.65       | 0.70     | 0.68     |
| GNN-CNN    | 0.69       | 0.71     | 0.70     |
| GNN        | 0.67       | 0.70     | 0.68     |
| GRU        | 0.64       | 0.68     | 0.65     |


## Datasets

### 1. IMDB Reviews — 50k movie reviews (25k/25k, binary sentiment).
https://www.tensorflow.org/datasets/catalog/imdb_reviews

### 2. Twitter (Sentiment140) — 1.6M tweets, positive/negative/neutral labels
[http://help.sentiment140.com/for-students](https://www.tensorflow.org/datasets/catalog/sentiment140?hl=ko)

### 3. SMS Spam Collection — 6k SMS messages, labeled spam/ham
https://archive.ics.uci.edu/datasets/sms+spam+collection

### 4. Classical English Literature, curated for annotation & style diversity
Shakespeare's Works: (https://www.opensourceshakespeare.org/)


| Dataset                               | Domain                    | Size           | Split                          | Purpose in Study                             | 
| ------------------------------------- | ------------------------- | -------------- | ------------------------------ | -------------------------------------------- | 
| Movie Reviews (IMDB / Social Network) | Sentiment (reviews)       | ~50,000        | 80/20 (stratified)             | Scenario embedding & topic–emotion alignment | 
| SMS Spam (UCI)                        | Short text classification | ~6,000         | 80/20 stratified               | Robustness on short/noisy text               | 
| Twitter Corpus (Sentiment140)         | Social media (tweets)     | 1.6M (subset)  | Time-aware 80/20               | Dynamic context & scenario transitions       | 
| Shakespeare Corpus                    | Literature                | Selected works | 80/20 (disjoint by work/scene) | Expression diversity & annotation patterns   | 


## How to Cite

If you use these datasets in your research, please cite the original authors and sources.

- **IMDB Reviews** — Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. Learning Word Vectors for Sentiment Analysis. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011), pp. 142–150.
  [Dataset link](https://ai.stanford.edu/~amaas/data/sentiment/)

- **SMS Spam Collection** — Almeida, T.A., Gómez Hidalgo, J.M., Yamakami, A. Contributions to the Study of SMS Spam Filtering: New Collection and Results. Proceedings of the 2011 ACM Symposium on Document Engineering (DOCENG'11), Mountain View, CA, USA, 2011.
  [UCI Repository link](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

- **Twitter Corpus (Sentiment140)** — Go, A., Bhayani, R. and Huang, L. (2009) Twitter Sentiment Classification Using Distant Supervision. CS224N Project Report, Stanford, 1-12.  
  [Sentiment140 dataset](http://help.sentiment140.com/for-students)

- **Shakespeare Corpus** — Public domain texts from *Open Source Shakespeare*.  
 [Open Source Shakespeare](https://www.opensourceshakespeare.org/)


### Visualization
Please refer to Fig.3–6 in the paper. 

## PDF Version
Due to confidentiality, please refer to the research paper for detailed information on the algorithm and methodology.
