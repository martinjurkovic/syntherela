import torch

# Please run `pip install -U sentence-transformers`
from sentence_transformers import SentenceTransformer
from torch import Tensor


class GloveTextEmbedding:
    def __init__(self, device: torch.device | None = None):
        self.model = SentenceTransformer(
            'sentence-transformers/average_word_embeddings_glove.6B.300d',
            device=device,
        )

    def __call__(self, sentences: list[str]) -> Tensor:
        return self.model.encode(sentences, convert_to_tensor=True)
