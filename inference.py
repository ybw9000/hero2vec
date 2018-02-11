import sys
import pickle

import numpy as np
import pandas as pd

from utils.prediction import *

def main():

    model_dir = './output/hero/model.p'
    model = pickle.load(open(model_dir, 'rb'))

    hero2ix_dir = './input/hero2ix.csv'
    hero2ix_df = pd.read_csv(hero2ix_dir, index_col=0)

    indicator = Predictor(model, hero2ix_df)

    heroes = sys.argv[1:]
    center_hero = indicator.predict(heroes)
    print('suggested hero: ', center_hero)

if __name__ == '__main__':
    main()
