# Predicting_Video_Games
ML models to predict the outcome of an Overwatch game

# Table of Contents
1. [Introduction](README.md#introduction)
2. [Challenges](README.md#challenges)
3. [Thoughts](README.md#Thoughts)
4. [Baseline model](README.md#baseline-model)

# Introduction
The goal of this project is to predict the final outcome (win/loss rate) of team in a video game, particularly multiplayer online games like Overwatch/Dota2/LoL, given the stats at a certain time in the game, like kills/deaths/team_compositions.

# Challenges

For this challenge, we're asking you to take an input file that lists campaign contributions by individual donors and distill it into two output files:

1. The gameplay is essentially a time-series, the model should be able to reason the win/loss rate not only by the current status of the game, but all the history that already happened in the game.

2. The features can be pretty limited, most common available features are simply just kills feed and team composition. This makes the info of the team composition very valuable, especially considering that the team composition actually plays an extremely import role in the game.

# Thoughts

1. A natural way to tackle the 1st challenge is by leveraging the RNN. However, the size of the data set can be limited. This brings difficulties in optimizing/training the RNN nets.

2. The characters/heroes can be simply modeled by one-hot encodings. However, even Overwatch already has 26 heroes and considering the enemy teams, this can be a 52 dimension sparse vector which is not favorable given a small data set. Yet, heroes can be viewed as words in a vocabulary and they certainly have lost of similarities as well, so they can be modeled by distributed vector representations like word2vec. A team composition is like a sentence, you have the center heroes and context heroes.

# Baseline model

A baseline model is simply via a Logistic Regression. The accuracy of it is expected to be 0.5 (random guess) in the beginning of the game when there is no useful input and climbs up later in the game with more meaningful game stats.
