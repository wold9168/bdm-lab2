import pandas as pd
import numpy as np


class Preprocess:
    def __init__(self):
        pass

    def delete_string_attr(self, df, ignore_list: list = []):
        cols_to_drop = []
        # ['Player', 'Team', 'Pos', 'Awards', 'Player-additional', 'player_name', 'team_abbreviation', 'college', 'country', 'draft_year', 'draft_round', 'draft_number', 'season']

        for i, types in enumerate(df.dtypes):
            if types == np.dtype("O"):
                if df.columns[i] not in ignore_list:
                    cols_to_drop.append(df.columns[i])
        df = df.drop(columns=cols_to_drop)  # df drop allthing about string type
        return df

    def fill_na(self, df):
        # 3P% 2P% eFG% FT% -> mean
        # Awards college -> We don't care, causal. It will be dropped when we call delete_string_attr()
        df["3P%"] = df["3P%"].fillna(df["3P%"].mean())
        df["2P%"] = df["2P%"].fillna(df["2P%"].mean())
        df["eFG%"] = df["eFG%"].fillna(df["eFG%"].mean())
        df["FT%"] = df["FT%"].fillna(df["FT%"].mean())
        return df

    def preprocess(
        self,
        player_stats_total_csv="./data/player_stats_total.csv",
        player_stats_body_csv="./data/player_stats_body.csv",
    ):
        player_stats_df = pd.read_csv(player_stats_total_csv).iloc[:-1]
        player_stats_body_df = pd.read_csv(player_stats_body_csv)
        df = pd.merge(
            player_stats_df,
            player_stats_body_df,
            left_on="Player",
            right_on="player_name",
        )
        df = df.drop_duplicates(subset=["Player", "Team"])
        df = self.fill_na(df)
        df = self.delete_string_attr(df, ["Player", "Team"])
        df["player_height"] = pd.to_numeric(df["player_height"], errors="coerce")
        df["player_weight"] = pd.to_numeric(df["player_weight"], errors="coerce")
        df["player_bmi"] = df["player_weight"] / ((df["player_height"] / 100) ** 2)
        numeric_cols = df.columns.drop(["Player", "Team"]).tolist()
        team_power = df.groupby("Team")[numeric_cols].mean().reset_index()
        return team_power
