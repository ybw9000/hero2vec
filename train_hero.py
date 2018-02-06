import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from model.hero2vec import *
from utils.evaluation import *
from utils.dataset import DataFrameIterator

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torch.utils.data import Dataset

def train(model, dataloader, loss_function=nn.CrossEntropyLoss(),
          init_lr=0.1, epochs=100, lr_decay_epoch = 30,
          print_epoch = 10, gpu=False):

    # Cuda is not critical for this task with low dimensionol inputs
    if gpu and torch.cuda.is_available():
        model.cuda()

    losses = []
    for epoch in range(epochs):

        # learning rate decay
        div, mod = divmod(epoch, lr_decay_epoch)
        if mod == 0:
            optimizer = optim.SGD(model.parameters(), lr=init_lr*(0.1)**div)

        total_loss = torch.Tensor([0])

        # iterate the dataset to load context heroes(team) and center hero(target)
        for teams, targets in dataloader:

            if gpu and torch.cuda.is_available():
                teams = teams.cuda()
                targets = targets.cuda()

            # wrap the embeddings of the team and target center hero to Variable
            inputs = autograd.Variable(teams)
            targets = autograd.Variable(targets.view(-1))

            # zero out the accumulated gradients
            model.zero_grad()

            # Run the forward pass
            out = model(inputs)

            # Compute your loss function.
            loss = loss_function(out, targets)

            # backpropagate and update the embeddings
            loss.backward()
            optimizer.step()

            # record total loss in this epoch
            total_loss += loss.cpu().data

        if epoch % print_epoch == 0:
            print('epoch: %d, loss: %.3f' % (epoch, total_loss/len(dataloader)))

        losses.append(total_loss/len(dataloader))
    # return losses for plot
    return np.array(losses)

def save_embeddings(model, filename = 'embeddings.npy'):
    embeddings = model.embeddings.weight.cpu().data.numpy()
    np.save(file = filename, arr=embeddings)

def main():

    data_dir = sys.argv[1]
    hero2ix_dir = sys.argv[2]

    # import DataFrame and hero2ix dictionary
    heroes_df = pd.read_csv(data_dir, index_col=0)
    hero2ix_df = pd.read_csv(hero2ix_dir, index_col=0)
    heroes_df = heroes_df.dropna().reset_index(drop=True)
    hero2ix = dict(zip(hero2ix_df.hero, hero2ix_df.ID))
    # heroes = hero2ix_df['hero'].values

    # train test split
    split = int(len(heroes_df)*0.9)
    heroes_train = heroes_df.iloc[:split]
    heroes_test = heroes_df.iloc[split:]

    # build dataset generator
    train_gen = DataFrameIterator(heroes_train, hero2ix)
    test_gen = DataFrameIterator(heroes_test, hero2ix)

    # Use Dataloader class in pytorch to generate batched data
    batch_size = 16
    loader_train = DataLoader(train_gen, batch_size=batch_size,
                              sampler=sampler.RandomSampler(train_gen),
                              num_workers=4)
    loader_test = DataLoader(test_gen, batch_size=batch_size,
                              sampler=sampler.SequentialSampler(test_gen),
                              num_workers=4)

    # define model, totally three models in hetor2vec.py
    model = CBOH(embedding_dim=10, heropool_size=len(hero2ix))

    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # run train
    losses = train(model=model, dataloader=loader_train, loss_function=loss_function,
                   init_lr=0.1, epochs=20, lr_decay_epoch=8, print_epoch=2, gpu=False)

    # check test accuracy
    print('accuracy: ', accuracy(model, dataloader=loader_test,
                                 batch_size=batch_size, gpu=False))

    # save embeddings as numpy arrays
    output_dir = './output/hero/hero_embeddings.npy'
    save_embeddings(model, filename=output_dir)

    # plot loss vs epoch
    plot_loss(losses, './output/hero/loss_hitory.png')

    # project embeddings to 2d plane
    plot_embeddings(model, hero2ix)

if __name__ == '__main__':
    main()
