# Graph-Based Cross-Domain Embedding Pipeline


A modular pipeline for building, training, and evaluating user–item embeddings across multiple domains using various GNNs.
This project constructs a heterogeneous bipartite graph from one or more Amazon datasets and trains GAT / GraphSAGE / GCN encoders to produce user and item embeddings.
Embeddings are exported for downstream tasks (recommendation, similarity search, clustering, diffusion models, etc.).


The pipeline includes: 
- Multi-dataset ingestion (target + multiple sources)
- End-to-end ETL (raw → cleaned → structured)
- Node and edge feature engineering
- Heterogenous graph construction using PyTorch geometric
- GAT / SAGE / GCN training with BPR loss
- Embedding export (Parquet)
- HR@K, Precision@K, Recall@K, NDCG@K evaluation

## Features

- Supports multiple raw datasets
- Builds a user-item bipartite HeteroData graph
- Configurable GAT, GraphSAGE, GCN encoders
- BPR pairwise ranking loss
- Parquet-based fast loading and saving
- Train/Val/Test edge-level split
- Exports embeddings as Parquet
- HR@K, NDCG@K evaluation metrics
- Fully config-driven

## Project Structure


```
Hetero-GNN/
│
├─ config/
│   └─ default.yaml                  # Global project configuration (datasets, model, training, logging)
│
├─ data/
│   ├─ raw/                          # Unprocessed JSONL datasets (magazine, appliances, movies, etc.)
│   └─ processed/                    # Pipeline outputs
│       ├─ interactions.parquet      # Cleaned user–item interactions
│       ├─ user_features.parquet     # Engineered user node features
│       ├─ item_features.parquet     # Engineered item node features
│       ├─ hetero_graph.pt           # PyG HeteroData graph (bipartite user–item graph)
│       ├─ gat_model.pt              # Trained GAT encoder weights
│       ├─ sage_model.pt             # Trained GraphSAGE encoder weights
│       ├─ gcn_model.pt              # Trained GCN encoder weights
│       ├─ user_embeddings.parquet   # Exported user embeddings
│       └─ item_embeddings.parquet   # Exported item embeddings
│
├─ src/
│   ├─ app/
│   │   ├─ config.py                 # Yaml-based configuration loader (dataclasses)
│   │   └─ logger.py                 # Colorful & file-based logging utility
│   │
│   ├─ data/
│   │   ├─ parse_raw.py              # Load raw datasets → clean interactions → map IDs/idx
│   │   ├─ split_interactions.py     # Train/val/test edge-level splitting
│   │   ├─ build_features.py         # User & item feature engineering
│   │   └─ build_graph.py            # Build hetero bipartite PyG graph
│   │
│   ├─ model/
│   │   ├─ gat.py                    # Heterogeneous GAT encoder (attention)
│   │   ├─ sage.py                   # Heterogeneous GraphSAGE encoder (sampling)
│   │   └─ gcn.py                    # Heterogeneous GCN encoder (message passing)
│   │
│   ├─ train/
│   │   ├─ train_gat.py              # Universal trainer for GAT/SAGE/GCN
│   │   └─ metrics.py                # HR@K, NDCG@K, Recall@K evaluation utilities
│   │
│   ├─ export/
│   │   └─ export_embeddings.py      # Export final embeddings as parquet
│   │
│   ├─ debug/
│   │   ├─ check_edge.py             # Verify edge correctness after graph build
│   │   ├─ graph_visualizer.py       # Plot user–item neighborhood subgraph
│   │   └─ visualize_embeddings_2d.py# t-SNE/UMAP embedding visualization
│   │
│   └─ main.py                       # Full pipeline executor (ETL → Train → Export)
│
└─ README.md                         # Project documentation
```

