import pandas as pd
import numpy as np
import inspect
from tqdm import tqdm
from datetime import datetime



games = pd.read_csv('data/games.csv')
pff = pd.read_csv('data/pffScoutingData.csv')
players = pd.read_csv('data/players.csv')
plays = pd.read_csv('data/plays.csv').tail(100)

print('~~~~~~~~~ Reading in week data ~~~~~~~~~~~~')
week = pd.concat(map(pd.read_csv, ['data/week1.csv', 'data/week2.csv', 'data/week3.csv', 'data/week4.csv', 'data/week5.csv', 'data/week6.csv', 'data/week7.csv', 'data/week8.csv']), ignore_index=True)
print('~~~~~~~~~ Read in week data ~~~~~~~~~~~~')
qbs = players.loc[players['officialPosition'] == "QB", "nflId"].to_list()


class PlayData():

    def __init__(self):
        self.col = np.array([member[0][11:] for member in inspect.getmembers(self, predicate=inspect.ismethod)[:-2]])

        def replace(arr, old, new):
            arr = np.insert(arr, np.where(arr == old)[0], new)
            arr = arr.flatten()
            return np.delete(arr, np.where(arr == old)[0])

        # replace with types of blockers
        self.col = replace(self.col, 'types_of_blockers', pff['pff_blockType'].unique()[1: ])



    def processPlay(self, gameId, playId):
        self.gameId = gameId
        self.playId = playId

        self.pff_playData = pff[(pff['gameId'] == gameId) & (pff['playId'] == playId)]
        self.meta_playData = plays[(plays['gameId'] == gameId) & (plays['playId'] == playId)]
        self.week_playData = week[(week['gameId'] == gameId) & (week['playId'] == playId)]

        feature = np.concatenate(tuple([value[1]() for value in inspect.getmembers(self, predicate=inspect.ismethod)[:-2]]), axis=None)
        return feature;

    def __numblockers(self):
        return self.pff_playData[pff['pff_role'] == "Pass Block"].shape[0]

    ## type of each blocker
    ## ['SW' 'PP' 'PT' 'CL' 'PA' 'PU' 'CH' 'NB' 'BH' 'UP' 'SR' 'PR']
    def __types_of_blockers(self):
        blockerTypes = pff['pff_blockType'].unique()[1: ]
        getNumBlockerType = np.vectorize(lambda type: self.pff_playData[pff['pff_blockType'] == type].shape[0])
        return getNumBlockerType(blockerTypes)

    def __pff_backFieldBlock(self):
        return self.pff_playData['pff_backFieldBlock'].dropna().shape[0]

    def __dropbackType(self):
        return self.meta_playData['dropBackType'].values[0]

    def __beatenByDefender(self):
        return int(self.pff_playData['pff_beatenByDefender'].sum() > 0)

    def __hitAllowed(self):
        return int(self.pff_playData['pff_hitAllowed'].sum() > 0)

    def __hurryAllowed(self):
        return int(self.pff_playData['pff_hurryAllowed'].sum() > 0)

    def __sackAllowed(self):
        return int(self.pff_playData['pff_sackAllowed'].sum() > 0)

    def __releaseTime(self):
        playEndEvents = ['pass_forward', 'fumble', 'qb_sack', 'qb_strip_sack']
        start = datetime.strptime(self.week_playData[self.week_playData['event'] == 'ball_snap']['time'].values[0], "%Y-%m-%dT%H:%M:%S.%f")
        endPlays = self.week_playData[self.week_playData['event'].isin(playEndEvents)]
        if endPlays.size == 0:
            return np.nan
        end = datetime.strptime(endPlays['time'].values[0], "%Y-%m-%dT%H:%M:%S.%f")

        return (end - start).total_seconds()

    def __offenseFormation(self):
        return self.meta_playData['offenseFormation'].values[0]

    def __playDirection(self):
        return self.week_playData['playDirection'].values[0]


if __name__ == '__main__':
    data = PlayData()
    compiled = []
    for index, play in tqdm(plays.iterrows(), total = plays.shape[0]):
        compiled.append(data.processPlay(play['gameId'], play['playId']))

    output = pd.DataFrame(data=compiled, columns=data.col)
    print(output)
    output.to_csv('data/features.csv')
