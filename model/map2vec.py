import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

class CBOM(nn.Module):

    def __init__(self, hero_embeddings, mappool_size):
        """
        Initialize an NN with one hidden layer.
        inputs:
            hero_embeddings: numpy array
            mappool_size: int
        """
        super().__init__()
        self.mappool_size = mappool_size
        self.hero_embeddings_data = hero_embeddings
        #initialize hero_embeddings from the numpy array
        self.heropool_size, self.hero_embedding_dim = hero_embeddings.shape
        self.hero_embeddings = nn.Embedding(self.heropool_size, self.hero_embedding_dim)

        # for one hidden layer the embedding_dim of map has to the same as heroes
        self.map_embedding_dim = self.hero_embedding_dim
        self.map_embeddings = nn.Embedding(self.mappool_size, self.map_embedding_dim)
        self.init_emb()

    def init_emb(self):
        """
        initialize with kaiming_normal
        """
        self.hero_embeddings.weight.data = torch.Tensor(self.hero_embeddings_data)
        init.kaiming_normal(self.map_embeddings.weight.data)

    def forward(self, inputs):
        """
        inputs:
            inputs: torch.autograd.Variable, size = (N, 6)
        returns:
            out: torch.autograd.Variable, size = (N, mappool_size)
        """
        # read all the embeddings out from map_embeddings
        indexes = autograd.Variable(torch.arange(0, self.mappool_size).long())
        hero_embeds = self.hero_embeddings(inputs).sum(dim=1)
        map_embeds = self.map_embeddings(indexes)
        out = torch.matmul(hero_embeds, map_embeds.t())
        return out

class CBOMTrilayer(nn.Module):
    
    def __init__(self, hero_embeddings, mappool_size, map_embedding_dim=10, hidden_dim=20):
        """
        Initialize an NN with three hidden layers.
        inputs:
            hero_embeddings: numpy array
            mappool_size: int
            map_embedding_dim: int
            hidden_dim: int
        """
        super().__init__()
        self.hero_embeddings_data = hero_embeddings
        self.mappool_size = mappool_size
        self.map_embedding_dim = map_embedding_dim
        self.hidden_dim = hidden_dim

        #initialize hero_embeddings from the numpy array
        self.heropool_size, self.hero_embedding_dim = hero_embeddings.shape
        self.hero_embeddings = nn.Embedding(self.heropool_size, self.hero_embedding_dim)

        self.linear1 = nn.Linear(self.hero_embedding_dim, self.hidden_dim)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(self.hidden_dim, self.map_embedding_dim)
        self.relu2 = nn.ReLU()
        self.map_embeddings = nn.Embedding(self.mappool_size, self.map_embedding_dim)
        self.init_emb()

    def init_emb(self):
        """
        initialize with kaiming_normal
        """
        self.hero_embeddings.weight.data = torch.Tensor(self.hero_embeddings_data)
        init.kaiming_normal(self.map_embeddings.weight.data)
        init.kaiming_normal(self.linear1.weight.data)
        self.linear1.bias.data.zero_()
        init.kaiming_normal(self.linear2.weight.data)
        self.linear2.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs:
            inputs: torch.autograd.Variable, size = (N, 6)
        returns:
            out: torch.autograd.Variable, size = (N, mappool_size)
        """
        # read all the embeddings out from map_embeddings
        indexes = autograd.Variable(torch.arange(0, self.mappool_size).long())
        hero_embeds = self.hero_embeddings(inputs).sum(dim=1)
        pipe = nn.Sequential(self.linear1, self.relu1, self.linear2, self.relu2)
        last = pipe(hero_embeds)
        # skip connection like resnet
        if self.hero_embedding_dim == self.map_embedding_dim:
            last = last + hero_embeds
        map_embeds = self.map_embeddings(indexes)
        out = torch.matmul(last, map_embeds.t())
        return out
