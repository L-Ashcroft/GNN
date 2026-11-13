"""Source/process conditioning; initialization is the identity FiLM map."""
import torch
from torch import nn


class DomainEmbedding(nn.Module):
    def __init__(self,num_domains,dimension):
        super().__init__();self.embedding=nn.Embedding(num_domains,dimension)
        nn.init.normal_(self.embedding.weight,mean=0.,std=.02)

    def forward(self,domain_ids):
        if domain_ids is None:raise ValueError('Domain-aware model requires explicit domain_ids')
        if (domain_ids<0).any() or (domain_ids>=self.embedding.num_embeddings).any():
            raise ValueError('Unknown/unregistered domain')
        return self.embedding(domain_ids)


class FiLM(nn.Module):
    def __init__(self,dimension,embedding_dim):
        super().__init__();self.to_gamma_beta=nn.Linear(embedding_dim,2*dimension)
        nn.init.zeros_(self.to_gamma_beta.weight);nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self,embedding):
        gamma,beta=self.to_gamma_beta(embedding).chunk(2,dim=-1)
        return 1.+gamma,beta
