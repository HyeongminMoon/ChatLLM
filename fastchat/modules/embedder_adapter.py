import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, List, Optional

import sys

if sys.version_info >= (3, 9):
    from functools import cache
else:
    from functools import lru_cache as cache
    
class Embedder():
    def __init__(self):
        pass
    
    def load(self):
        pass

    def embed(self, text: str) -> list[torch.Tensor]:
        pass
    
    def match(self, name: str):
        return True

embedder_adapters: List[Embedder] = []

def register_embedder(cls):
    embedder_adapters.append(cls())

@cache
def get_embedder(name: str) -> Embedder:
    for adapter in embedder_adapters:
        if adapter.match(name):
            adapter.load()
            return adapter

    raise ValueError(f"No valid model adapter for {name}")

# sentence transformers embedder for jhgan/ko-sroberta-multitask
class KoSrobertaEmbedder(Embedder):
    def __init__(self):
        self.model = None
    
    def load(self):
        if self.model is None:
            self.model = SentenceTransformer("jhgan/ko-sroberta-multitask")
            self.embed = self.model.encode
        
    def match(self, name):
        return 'sroberta' in name.lower() and 'ko' in name.lower()
    
        
# sentence transformers embedder for sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja
class KoMiniLMEmbedder(Embedder):
    def __init__(self):
        self.model = None
    
    def load(self):
        if self.model is None:
            self.model = SentenceTransformer("sangmini/msmarco-cotmae-MiniLM-L12_en-ko-ja")
            self.embed = self.model.encode
        
    def match(self, name):
        return 'minilm' in name.lower() and 'ko' in name.lower()
        
# sentence transformers embedder for ddobokki/klue-roberta-base-nli-sts-ko-en
class KoKluerobertaEmbedder(Embedder):
    def __init__(self):
        self.model = None
    
    def load(self):
        if self.model is None:
            self.model = SentenceTransformer("ddobokki/klue-roberta-base-nli-sts-ko-en")
            self.embed = self.model.encode
        
    def match(self, name):
        return 'klue-roberta' in name.lower() and 'ko' in name.lower()


register_embedder(KoSrobertaEmbedder)
register_embedder(KoMiniLMEmbedder)
register_embedder(KoKluerobertaEmbedder)