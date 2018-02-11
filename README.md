# Hero2Vec
A Machine learning model to understand the game design and player experience of a video game, the Overwatch.

# Table of Contents
1. [Introduction](README.md#introduction)
2. [Challenges](README.md#challenges)
3. [Motivations](README.md#motivations)
4. [Model Selection](README.md#model-selection)
5. [Model Architecture](README.md#model-architecture)
6. [Input](README.md#input)
7. [Output](README.md#output)
8. [Repo Structure](README.md#repo-structure)
9. [Setup](README.md#setup)
10. [Usage](README.md#usage)

# Introduction

The goal of this project is to predict the outcome (winning-rate) of a team in a video game, particularly multiplayer online games like Overwatch/Dota2/LoL, given the status at a certain time in the game, like kills/deaths/team-compositions, etc.

This repo focuses on part of the project, namely, modeling the team compositions (or the heroes) and maps in the game.

# Challenges

1. The dataset is not large enough. We only have the results from less than 300 games.

2. The dataset consist of a lot of categorical features, like team compositions and maps. A simple one-hot encoding can result in high dimensional sparse input and unfortunately we don't have enough data to conquer the **curse of dimensionality**. Moreover, the team-composition/map plays an extremely important role in the game, so we can't simply drop it.

# Motivations

1. Just like humans (or words), heroes have their own characteristics and also share some **similarities**. So rather than one-hot orthogonal vectors, they can be represented by **distributed representations**. What's more, just like words in a sentence, heroes in a team also have strong **co-occurrence**. So heroes can be modeled in a very similar fashion as **word2vec**, i.e., **hero2vec**.

2. The team compositions are widely available online. This is independent of my own dataset and can serve the training of hero2vec just like Wiki corpus used for word2vec.

3. By modeling heroes in the game by distributed representations, I can not only address the curse of dimensionality, but also gain valuable information on the game designs of the heroes as well as the how the players appreciate these designs.

4. All the above motivations apply to the maps similarly, i.e., **map2vec**.

# Model Selection

1. As mentioned above, heroes in a team have strong co-occurrence, i.e., the conditional probability P(h1|h2.., h6) (6 heroes in a team) is high. h1 doesn't have to be a specific hero, any hero in the team can be this center hero. This is very suitable for the **Continuous Bag of Words (CBOW)** model, since the attributes of a team (or the 5 context heroes) are really a sum of the attributes of all the individuals, unlike the sum of context words in a sentence is not always intuitive.

2. The map in the game can be modeled in a similar way. The conditional probability P(map|team) is high. So the weight of the last affine layer of the classifier is the embeddings for the maps.

# Model Architecture

1. hero2vec. The model pipeline is as follows:
`input context heroes (5 heroes)` -> `embeddings` -> `sum` -> `fully connected layers` -> `softmax (center hero)`

2. map2vec. The model pipeline is as follows:
`input team (6 heroes)` -> `hero2vev embeddings` -> `sum` -> `fully connected layers` -> `map embeddings` -> `softmax (map)`

# Input

1. `teams.csv` under `input` folder. This is a csv table that contains the team composition. Can be easily changed to other team-based games like Dota2/LoL.

2. `map_teams.csv` under `input` folder. This the csv table that contains both the team and map composition.

3. `hero2ix.csv` under `input` folder. This is csv table that maps the input hero names to their int ID and further to the embeddings. Can be easily customized in case different name is used for the same hero, e.g., 'dva' (used in this one) is written as 'D.Va'.

4. `map2ix.csv` under `input` folder. This is csv table that maps the input map names to their int ID and further to the embeddings.

# Output

1. `hero` folder
Output contains a graph showing the embeddings (after PCA to 2D) of the heroes `hero_embeddings_2d.png`, a numpy array contains the embeddings `hero_embeddings.npy`, a graph of the training loss `loss_history.png` and pickled model `model.p`. For example, the `hero_embeddings_2d.png` looks like:

<img src="https://github.com/ybw9000/hero2vec/blob/master/output/hero/hero_embddings_2d.png" align="center">

2. `map` folder
Output contains a graph showing the embeddings (after PCA to 2D) of the maps `map_embeddings_2d.png`, a numpy array contains the embeddings `map_embeddings.npy` and a graph of the training loss `loss_history.png`. For example, the `map_embeddings_2d.png` looks like:

<img src="https://github.com/ybw9000/hero2vec/blob/master/output/map/map_embddings_2d.png" align="center">

# Repo Structure

The directory structure for the repo looks like this:

    ├── README.md
    ├── train_hero.py
    ├── train_map.py
    ├── inference.py
    ├── setup
    │   └── install.sh
    │   └── install_without_torch.sh
    ├── model
    │   └── __init__.py    
    │   └── hero2vec.py
    │   └── map2vec.py
    ├── utils
    │   └── __init__.py
    │   └── dataset.py
    │   └── evaluation.py
    │   └── prediction.py
    ├── input
    │   └── hero2ix.csv
    │   └── map2ix.csv
    │   └── teams.csv
    │   └── map_teams.csv
    └── output
        ├── hero
        │   └── hero_embeddings_2d.png
        │   └── hero_embeddings.npy
        │   └── loss_history.npy
        │   └── model.p
        └── map
            └── map_embeddings_2d.png
            └── map_embeddings.npy
            └── loss_history.npy
# Setup

Under `setup` folder, run:

`bash install.sh`

if issues occurs with installing pytorch, please refer to http://pytorch.org/ for installation of pytorch. Then run:

`bash install_without_torch.sh`

# Usage

1. Train hero2vec. run: `python train_hero.py ./input/teams.csv ./input/hero2ix.csv`

2. Train map2vec. run: `python train_map.py ./input/map_teams.csv ./input/hero2ix.csv ./input/map2ix.csv`

3. Predict the center hero given five other heroes. run: `python inference.py <heroes>`. `<heroes>` contains the hero names of five known members. For example: `python inference.py dva genji tracer lucio winston`. Note: hero names must be in the `hero` column in `hero2ix.csv` in `input` folder.
