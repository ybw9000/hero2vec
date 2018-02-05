import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import init

class CBOH(nn.Module):

    def __init__(self, heropool_size, embedding_dim):
        """
        Initialize an NN with one hidden layer. Weight of the hidden layer is
        the embedding.
        inputs:
            heropool_size: int
            embedding_dim: int
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(heropool_size, embedding_dim)
        self.affine = nn.Linear(embedding_dim, heropool_size)
        self.init_emb()

    def init_emb(self):
        """
        init embeddings and affine layer
        """
        initrange = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        self.affine.weight.data.uniform_(-0, 0)
        self.affine.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs:
            inputs: torch.autograd.Variable, size = (N, 5)
        returns:
            out: torch.autograd.Variable, size = (N, heropool_size)
        """
        embeds = self.embeddings(inputs).sum(dim=1) #contiuous
        out = self.affine(embeds)
        return out

class CBOHBilayer(nn.Module):

    def __init__(self, heropool_size, embedding_dim, hidden_dim=10):
        """
        Initialize an NN with two hidden layers. Weight of the first hidden
        layer is the embedding.
        inputs:
            heropool_size: int
            embedding_dim: int
            hidden_dim: int
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(heropool_size, embedding_dim)
        #Initialize 2nd hidden layer with dimension = hidden_dim
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        self.affine = nn.Linear(hidden_dim, heropool_size)
        self.init_emb()

    def init_emb(self):
        """
        init embeddings and affine layer. The weight of the 2nd hidden layer is
        initialized by Kaiming_norm.
        """
        initrange = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        init.kaiming_normal(self.linear1.weight.data)
        self.linear1.bias.data.zero_()
        self.affine.weight.data.uniform_(-0, 0)
        self.affine.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs:
            inputs: torch.autograd.Variable, size = (N, 5)
        returns:
            out: torch.autograd.Variable, size = (N, heropool_size)
        """
        embeds = self.embeddings(inputs).sum(dim=1) #contiuous
        pipe = nn.Sequential(self.linear1, self.relu1, self.affine)
        out = pipe(embeds)
        return out

class CBOHTrilayer(nn.Module):

    def __init__(self, heropool_size, embedding_dim, hidden_dim=10,
                 affine_dim=10):
        """
        Initialize an NN with three hidden layers. Weight of the first hidden
        layer is the embedding.
        inputs:
            heropool_size: int
            embedding_dim: int
            hidden_dim: int
            affine_dim: int
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.affine_dim = affine_dim
        self.embeddings = nn.Embedding(heropool_size, embedding_dim)
        #Initialize 2nd hidden layer with dimension = hidden_dim
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        #Initialize 3rd hidden layer with dimension = affine_dim
        self.linear2 = nn.Linear(hidden_dim, affine_dim)
        self.relu2 = nn.ReLU()
        self.affine = nn.Linear(affine_dim, heropool_size)
        self.init_emb()

    def init_emb(self):
        """
        init embeddings and affine layer. The weights of the 2nd and 3rd hidden
        layers are initialized by Kaiming_norm.
        """
        initrange = 0.5 / self.embedding_dim
        self.embeddings.weight.data.uniform_(-initrange, initrange)
        init.kaiming_normal(self.linear1.weight.data)
        self.linear1.bias.data.zero_()
        init.kaiming_normal(self.linear2.weight.data)
        self.linear2.bias.data.zero_()
        self.affine.weight.data.uniform_(-0, 0)
        self.affine.bias.data.zero_()

    def forward(self, inputs):
        """
        inputs:
            inputs: torch.autograd.Variable, size = (N, 5)
        returns:
            out: torch.autograd.Variable, size = (N, heropool_size)
        """
        embeds = self.embeddings(inputs).sum(dim=1)
        pipe = nn.Sequential(self.linear1, self.relu1, self.linear2, self.relu2)
        # skip connection to assist gradient flow
        if self.embedding_dim == self.affine_dim:
            out = self.affine(pipe(embeds) + embeds)
        else:
            out = self.affine(pipe(embeds))
        return out
