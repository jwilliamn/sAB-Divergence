import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import torch


def ll_gaussian(y, mu, log_var):
    sigma = torch.exp(0.5 * log_var)
    return -0.5 * torch.log(2 * np.pi * sigma**2) - (1 / (2 * sigma**2))* (y-mu)**2

def elbo(y_pred, y, mu, log_var):
    # likelihood of observing y given Variational mu and sigma
    likelihood = ll_gaussian(y, mu, log_var)
    
    # prior probability of y_pred
    log_prior = ll_gaussian(y_pred, 0, torch.log(torch.tensor(1.)))
    
    # variational probability of y_pred
    log_p_q = ll_gaussian(y_pred, mu, log_var)
    
    # by taking the mean we approximate the expectation
    return (likelihood + log_prior - log_p_q).mean()

def det_loss(y_pred, y, mu, log_var):
    return -elbo(y_pred, y, mu, log_var)
    
    
def log_norm(x, mu, std):
    """Compute the log pdf of x,
    under a normal distribution with mean mu and standard deviation std."""
    
    return -0.5 * torch.log(2*np.pi*std**2) -(0.5 * (1/(std**2))* (x-mu)**2)
    
 
 
class AB(torch.nn.Module):
    def __init__(self, alpha, beta, latent_dim=100):
        super(AB, self).__init__()
        self.n_latent = latent_dim # Number of latent samples
        self.softplus = torch.nn.Softplus()
        self.alpha = alpha
        self.beta = beta
        
        #The parameters we adjust during training.
        self.qm = torch.nn.Parameter(torch.randn(1,1).double(), requires_grad=True)
        self.qs = torch.nn.Parameter(torch.randn(1,1).double(), requires_grad=True)
        
        #create holders for prior mean and std, and likelihood std.
        self.prior_m = torch.randn(1,1).double()
        self.prior_s = torch.randn(1,1).double()
        self.likelihood_s = torch.DoubleTensor((1))
        
        #Set the prior and likelihood moments.
        self.prior_s.data.fill_(1.0)
        self.prior_m.data.fill_(0.9)
        self.likelihood_s.data.fill_(5.5)

        
    def generate_rand(self):
        return np.random.normal(size=(self.n_latent,1))
    
    def reparam(self, eps):
        eps = Variable(torch.DoubleTensor(eps))
        return  eps.mul(self.softplus(self.qs)).add(self.qm)
    
    def compute_bound(self, x, t):
        eps = self.generate_rand()
        z = self.reparam(eps)
        q_likelihood = log_norm(z, self.qm, self.softplus(self.qs))
        prior = log_norm(z, self.prior_m, self.prior_s)
        likelihood = torch.sum(log_norm(t, x*z.transpose(0,1), self.likelihood_s), 0, keepdim=True).transpose(0,1)
        
        
        r1 = (self.alpha + self.beta)*(prior + likelihood) - q_likelihood
        c1 = torch.max(r1)
        exp1 = torch.exp((r1-c1))
        
        r2 = (self.alpha + self.beta -1)* q_likelihood
        c2 = torch.max(r2)
        exp2 = torch.exp((r2-c2))
        
        r3 = self.beta * (prior + likelihood) - (1- self.alpha) * q_likelihood
        c3 = torch.max(r3)
        exp3 = torch.exp((r3-c3))

        loss1 = (torch.log(torch.mean(exp1)) + c1) / ((self.beta + self.alpha) * self.alpha ) 
        loss2 = (torch.log(torch.mean(exp2)) + c2) / ((self.beta + self.alpha) * self.beta  ) 
        loss3 = (torch.log(torch.mean(exp3)) + c3) / (self.beta * self.alpha ) 
       
        return loss1 + loss2 + loss3
    

class KL(torch.nn.Module):
    def __init__(self, latent_dim=100):
        super(KL, self).__init__()
        self.n_latent = latent_dim # Number of latent samples
        self.softplus = torch.nn.Softplus()
        
        #The parameters we adjust during training.
        self.qm = torch.nn.Parameter(torch.randn(1,1).double(), requires_grad=True)
        self.qs = torch.nn.Parameter(torch.randn(1,1).double(), requires_grad=True)
        
        #create holders for prior mean and std, and likelihood std.
        self.prior_m = torch.randn(1,1).double()
        self.prior_s = torch.randn(1,1).double()
        self.likelihood_s = torch.DoubleTensor((1))
        
        #Set the prior and likelihood moments.
        self.prior_s.data.fill_(1.0)
        self.prior_m.data.fill_(0.9)
        self.likelihood_s.data.fill_(5.5)

        
    def generate_rand(self):
        return np.random.normal(size=(self.n_latent,1))
    
    def reparam(self, eps):
        eps = Variable(torch.DoubleTensor(eps))
        return  eps.mul(self.softplus(self.qs)).add(self.qm)
    
    def compute_bound(self, x, t):
        eps = self.generate_rand()
        z = self.reparam(eps)
        q_likelihood = log_norm(z, self.qm, self.softplus(self.qs))
        prior = log_norm(z, self.prior_m, self.prior_s)
        likelihood = torch.sum(log_norm(t, x*z.transpose(0,1), self.likelihood_s), 0, keepdim=True).transpose(0,1)
        
        
        r1 = prior + likelihood -q_likelihood
        c1 = torch.max(r1)
        exp1 = torch.exp((r1-c1))
        
        KL_div = (torch.log(torch.mean(exp1)) + c1)
       
        return KL_div
        

class Renyi(torch.nn.Module):
    def __init__(self, alpha, latent_dim=100):
        super(Renyi, self).__init__()
        self.n_latent = latent_dim # Number of latent samples
        self.softplus = torch.nn.Softplus()
        self.alpha = alpha
        
        #The parameters we adjust during training.
        self.qm = torch.nn.Parameter(torch.randn(1,1).double(), requires_grad=True)
        self.qs = torch.nn.Parameter(torch.randn(1,1).double(), requires_grad=True)
        
        #create holders for prior mean and std, and likelihood std.
        self.prior_m = torch.randn(1,1).double()
        self.prior_s = torch.randn(1,1).double()
        self.likelihood_s = torch.DoubleTensor((1))
        
        #Set the prior and likelihood moments.
        self.prior_s.data.fill_(1.0)
        self.prior_m.data.fill_(0.9)
        self.likelihood_s.data.fill_(5.5)

        
    def generate_rand(self):
        return np.random.normal(size=(self.n_latent,1))
    
    def reparam(self, eps):
        eps = Variable(torch.DoubleTensor(eps))
        return  eps.mul(self.softplus(self.qs)).add(self.qm)
    
    def compute_bound(self, x, t):
        eps = self.generate_rand()
        z = self.reparam(eps)
        q_likelihood = log_norm(z, self.qm, self.softplus(self.qs))
        prior = log_norm(z, self.prior_m, self.prior_s)
        likelihood = torch.sum(log_norm(t, x*z.transpose(0,1), self.likelihood_s), 0, keepdim=True).transpose(0,1)
        
        r1 = (1-self.alpha)*(prior + likelihood -q_likelihood)
        c1 = torch.max(r1)
        exp1 = torch.exp((r1-c1))

        renyi_div = (torch.log(torch.mean(exp1))+c1)/(1-self.alpha)
        return renyi_div