import torch
from torch.autograd import Variable

class Predictor():

    def __init__(self, model, hero2ix_df):
        """
        input:
            model: pytorch model
            hero2ix_df: pandas DataFrame
        """
        self.model = model
        self.model.eval()
        self.hero2ix_df = hero2ix_df

    def predict(self, heroes):
        """
        input:
            heroes: list of str
        return:
            center_hero: str
        """
        assert len(heroes) == 5, 'Input has to be 5 five heroes'

        for hero in heroes:
            if hero not in self.hero2ix_df.hero.values:
                raise KeyError('wrong hero name:' + hero)

        # find idxs for heroes
        team_idxs = list(self.hero2ix_df[self.hero2ix_df.hero.isin(heroes)].ID)

        inputs = Variable(torch.LongTensor(team_idxs)).view(-1, 5)
        out = self.model(inputs)
        val, idx = torch.max(out, dim=1)

        # map hero id to hero name
        center_hero = self.hero2ix_df.hero.loc[int(idx)]
        return center_hero
