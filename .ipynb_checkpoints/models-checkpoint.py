import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import torch


class VI_base(nn.Module):
    def __init__(self, input=1, hidden=10):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(input, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(input, hidden*2),
            nn.ReLU(),
            nn.Linear(hidden*2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var
        
        
class VI(nn.Module):
    def __init__(self, n_input=1, n_hidden=10, n_output=1):
        super().__init__()

        self.q_mu = nn.Sequential(
            nn.Linear(n_input, n_hidden*2),
            nn.ReLU(),
            nn.Linear(n_hidden*2, n_output),
            nn.ReLU()
        )
        self.q_log_var = nn.Sequential(
            nn.Linear(n_input, n_hidden*2),
            nn.ReLU(),
            nn.Linear(n_hidden*2, n_output),
            nn.ReLU()
        )

    def reparameterize(self, mu, log_var):
        # std can not be negative, thats why we use log variance
        sigma = torch.exp(0.5 * log_var) + 1e-5
        eps = torch.randn_like(sigma)
        return mu + sigma * eps

    def forward(self, x):
        mu = self.q_mu(x)
        log_var = self.q_log_var(x)
        return self.reparameterize(mu, log_var), mu, log_var