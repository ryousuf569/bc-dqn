"""
Synthetic data generator based on nba_api.
Randomly selects an NBA game and downloads its play-by-play.
"""

import random
import pandas as pd
import sqlite3
from nba_api.stats.endpoints import leaguegamefinder, playbyplayv3

db_path = 'real_games.db'
conn = sqlite3.connect(db_path)

''' games_df = leaguegamefinder.LeagueGameFinder(season_nullable='2025-26', 
                                             season_type_nullable='Regular Season').get_data_frames()[0]


unique_games = games_df[['GAME_ID', 'GAME_DATE', 'MATCHUP']].drop_duplicates()

for _ in range(500):
    random_game = unique_games.sample(1).iloc[0]
    game_id = random_game['GAME_ID']
    unique_games = unique_games[unique_games['GAME_ID'] != game_id]

    pbp_df = playbyplayv3.PlayByPlayV3(game_id=game_id, start_period=4,end_period=4).get_data_frames()[0]
    pbp_df = pbp_df[['gameId', 'clock', 'period', "shotDistance", "shotResult", "isFieldGoal", "scoreHome", "scoreAway",
                    "pointsTotal", "location", "actionType",]]

    pbp_df = pbp_df[pd.to_timedelta(pbp_df['clock']) <= pd.Timedelta(minutes=2)]
    score_diff = abs(pd.to_numeric(pbp_df['scoreHome'], errors='coerce').fillna(0).iloc[0]
    - pd.to_numeric(pbp_df['scoreAway'], errors='coerce').fillna(0).iloc[0])

    if score_diff <= 5:
        pbp_df.to_sql("real_games", conn, if_exists="append", index=False) '''

df = pd.read_sql('SELECT * FROM real_games', con=conn)
matrices = []

for _, game_df in df.groupby("gameId"):
    transitions = game_df["actionType"].tolist()

    game_matrix = pd.crosstab(
        pd.Series(transitions[:-1], name='From'),
        pd.Series(transitions[1:], name='To'),
        normalize='index'
    )

    matrices.append(game_matrix)
    print(game_matrix)
