import pandas as pd
import numpy as np
import json

plays_all = pd.read_csv('rawdata/plays.csv')
games_all = pd.read_csv('rawdata/games.csv')
players_all = pd.read_csv('rawdata/players.csv')
tackles_all = pd.read_csv('newdata/tackles_all_n.csv')

def game(gameId):
    game = games_all[games_all['gameId'] == gameId]
    return game

def tacklers(gameId, playId):
    tackles = tackles_all[(tackles_all['gameId'] == gameId) & (tackles_all['playId'] == playId)]
    tacklers = tackles.nflId.tolist()
    return tacklers

def carrier(gameId, playId):
    carrier = plays_all[(plays_all['gameId'] == gameId) & (plays_all['playId'] == playId)].ballCarrierId.item()
    return carrier

def tackles(gameId, playId):
    tackles = tackles_all[(tackles_all['gameId'] == gameId) & (tackles_all['playId'] == playId)]
    return tackles

def play(gameId, playId):
    play_df = plays_all[(plays_all['gameId'] == gameId) & (plays_all['playId'] == playId)]
    return play_df

def tracking(gameId, playId):
    df = pd.read_csv(f'newdata/tracking_game/{gameId}.csv')
    tracking_df = df[df['playId'] == playId]
    tracking_df = tracking_df.reset_index(drop = True)
    return tracking_df

def name(nflId):
    name = players_all[players_all['nflId'] == nflId]['displayName'].item()
    return name