import numpy as np
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import datasets

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch import nn
import torch


def train(model, optimizer, criterion, Xtrain_loader, ytrain_loader, epochs=3000):
    train_losses = []
    validation_losses = []

    for epoch in range(epochs):

        m.train()
        for x, y in zip(Xtrain_loader, ytrain_loader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            y_pred, mu, log_var = m(x)
            loss = -criterion.compute_bound(y_pred, y)
            loss.backward()
            optimizer.step()

            # save loss
            train_losses.append(loss.item())

        m.eval()
        for x, y in zip(Xval_loader, yval_loader):
            y_pred, mu, log_var = m(x)
            loss = -criterion.compute_bound(y_pred, y)

            # save loss
            validation_losses.append(loss.item())

        if epoch%500==0:
            #print('Epoch {:02d}/{} || Loss:  Train {:.4f} | Validation {:.4f}'.format(epoch, epochs, train_losses[-1], validation_losses[-1]))
            print(criterion.qm.data.numpy(), (criterion.softplus(criterion.qs).data**2).numpy())