import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import LowRankMultivariateNormal


class Encoder(nn.Module):
    def __init__(self, num_topics, hidden_size, vocab_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(vocab_size, hidden_size, False),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_size, num_topics),
            nn.Softmax(dim=1),
        )

        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, x):
        return self.layers(x)


class Generator(nn.Module):
    def __init__(self, num_topics, hidden_size, vocab_size
                 ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_topics, hidden_size, False),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_size, vocab_size),
            nn.Softmax(dim=1),
        )

        self.num_topics = num_topics
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def forward(self, z):
        return self.layers(z)

    @property
    def topic_word_dist(self):
        z = torch.eye(self.num_topics).to(next(self.parameters()).device)
        return self.layers(z)


class Discriminator(nn.Module):
    def __init__(self, num_topics, hidden_size, vocab_size):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(num_topics + vocab_size, hidden_size, False),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(0.1, True),
            nn.Linear(hidden_size, 1),
        )
        self.l1 =nn.Linear(num_topics + vocab_size, hidden_size, False)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.lr = nn.LeakyReLU(0.1, True)
        self.l2 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        x= self.l1(x)
        x = self.bn(x)
        x = self.lr(x)
        x = self.l2(x)
        return  x
        return self.layers(x)


class GaussianGenerator(nn.Module):
    def __init__(self, num_topics, word_vectors, vocab_size):
        super().__init__()

        assert vocab_size == word_vectors.shape[0]
        self.word_vectors = word_vectors.unsqueeze(1)  # -> (vocab_size, 1, vector_size)

        vocab_size, vector_size = word_vectors.shape

        mu = nn.Parameter(torch.randn(num_topics, vector_size))
        cov_factor = nn.Parameter(torch.randn(num_topics, vector_size, 1))
        cov_diag = nn.Parameter(torch.rand(num_topics, vector_size) * 50)

        self.register_parameter('mu', mu)
        self.register_parameter('cov_factor', cov_factor)
        self.register_parameter('cov_diag', cov_diag)
        self.clamp_cov_diag()

        self.num_topics = num_topics
        self.vocab_size = vocab_size

    def forward(self, x):
        multi_normal = LowRankMultivariateNormal(self.mu, self.cov_factor, self.cov_diag)
        topic_word_dist = multi_normal.log_prob(self.word_vectors).t()

        topic_word_dist = F.softmax(topic_word_dist, dim=-1)
        return torch.mm(x, topic_word_dist)

    def clamp_cov_diag(self, min_val=1e-6):
        with torch.no_grad():
            self.cov_diag.clamp_(min_val)

    @property
    def topic_word_dist(self):
        z = torch.eye(self.num_topics).to(next(self.parameters()).device)
        x = self(z)
        return x

    def extra_repr(self):
        return f'in_features={self.num_topics}, out_features={self.vocab_size}'
