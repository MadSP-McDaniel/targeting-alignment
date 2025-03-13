from .embeddings import EmbeddingLens, KVLens, TruncatedEmbeddingLens
from .lenses import Lens


# Custom Lenses
class LastEmbeddingLens(EmbeddingLens):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, layers=[-1], positions=[-1], **kwargs)


class LongitudinalLastEmbeddingLens(EmbeddingLens):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, positions=[-1], **kwargs)


class KeyLens(KVLens):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, type="key", **kwargs)


class ValueLens(KVLens):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, type="value", **kwargs)
