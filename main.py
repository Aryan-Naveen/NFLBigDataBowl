import pandas as pd
import numpy as np


games = pd.read_csv('data/games.csv')
pff = pd.read_csv('data/pffScoutingData.csv')
players = pd.read_csv('data/players.csv')
plays = pd.read_csv('data/players.csv')

def get_numblockers(gameIdPlayId):
    print(pff.head(10))

if __name__ == '__main__':
    get_numblockers()
