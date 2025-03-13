mkdir -p /workspace/data/embeddings
export DATA_DIR=/workspace/data/embeddings

huggingface-cli download jcnf0/targeting-alignment --repo-type dataset --local-dir data/embeddings