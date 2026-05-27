# scripts/dev/test_bge_m3.py

from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=False,
)

texts = [
    "The Tower of Babel is a story about language, ambition, and human division.",
    "A podcast transcript can be transformed into a dialogue between two speakers.",
    "Supply chain policy affects pharmaceutical companies entering the United States.",
]

outputs = model.encode(
    texts,
    batch_size=4,
    max_length=1024,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False,
)

embeddings = outputs["dense_vecs"]

print(type(embeddings))
print(embeddings.shape)
print(embeddings[0][:10])