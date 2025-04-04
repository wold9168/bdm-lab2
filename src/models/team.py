from .player import *
import pandas as pd
import numpy as np
from typing import *
import csv


class Team:
    def __init__(self, name: str, name_abbr: str):
        self.name = name
        self.name_abbr = name_abbr
        self.players: List[Player] = []

    def add_player(self, player: Player):
        self.players.append(player)

    def get_features(self) -> np.array:
        features = []
        for player in self.players:
            features.extend(player.features)
        return np.array(features)

    def update(self, team):
        if self.name != team.name or self.name_abbr != team.name_abbr:
            raise ValueError("Team name and abbreviation must match")
        self.players = team.players.copy()


class AllTeam:
    def __init__(self):
        self.teamlist: List[Team] = [
            Team("Atlanta Hawks", "ATL"),
            Team("Boston Celtics", "BOS"),
            Team("Brooklyn Nets", "BRK"),
            Team("Chicago Bulls", "CHI"),
            Team("Charlotte Hornets", "CHO"),
            Team("Cleveland Cavaliers", "CLE"),
            Team("Dallas Mavericks", "DAL"),
            Team("Denver Nuggets", "DEN"),
            Team("Detroit Pistons", "DET"),
            Team("Golden State Warriors", "GSW"),
            Team("Houston Rockets", "HOU"),
            Team("Indiana Pacers", "IND"),
            Team("Los Angeles Clippers", "LAC"),
            Team("Los Angeles Lakers", "LAL"),
            Team("Memphis Grizzlies", "MEM"),
            Team("Miami Heat", "MIA"),
            Team("Milwaukee Bucks", "MIL"),
            Team("Minnesota Timberwolves", "MIN"),
            Team("New Orleans Pelicans", "NOP"),
            Team("New York Knicks", "NYK"),
            Team("Oklahoma City Thunder", "OKC"),
            Team("Orlando Magic", "ORL"),
            Team("Philadelphia 76ers", "PHI"),
            Team("Phoenix Suns", "PHO"),
            Team("Portland Trail Blazers", "POR"),
            Team("Sacramento Kings", "SAC"),
            Team("San Antonio Spurs", "SAS"),
            Team("Toronto Raptors", "TOR"),
            Team("Utah Jazz", "UTA"),
            Team("Washington Wizards", "WAS"),
        ]

    def update_team(self, team: Team):
        for curteam in teamlist:
            if curteam.name == team.name and curteam.name_abbr == team.name_abbr:
                curteam.update_team(team)
                break
        else:
            raise ValueError(f"Team {team.name} not found")

    def get_team_abbr_with_name(self, name) -> Team:
        for team in self.teamlist:
            if team.name == name or team.name_abbr == name:
                return team.name_abbr
        raise ValueError(f"Team {name} not found")

    def get_team_short_name(self, team: Team):
        return team.name_abbr

    def get_team_short_name(self, name: str):
        for team in self.teamlist:
            if team.name == name or team.name_abbr == name:
                return team.name_abbr

    def get_all_teams(self) -> List[Team]:
        return self.teamlist

    def get_team_with_name(self, name) -> Team:
        for team in self.teamlist:
            if team.name == name or team.name_abbr == name:
                return team
        raise ValueError(f"Team {name} not found, type {type(name)}")


class TeamVSTeam:
    def __init__(self):
        self.History = {}
        self.teamabbr_index = AllTeam()

    def add_history(self, team1: Team, team2: Team, score: int):
        self.History.update({(team1.name_abbr, team2.name_abbr): score})

    def add_history(self, team1name_abbr: str, team2name_abbr: str, score: int):
        self.History.update({(team1name_abbr, team2name_abbr): score})

    # def get_history(self, team1: Team, team2: Team):
    #     return self.History.get((team1.name_abbr, team2.name_abbr))

    def get_history(self, team1_name: str, team2_name: str):
        return self.History.get(
            (
                self.teamabbr_index.get_team_abbr_with_name(team1_name),
                self.teamabbr_index.get_team_abbr_with_name(team2_name),
            )
        )

    def read_and_process_csv(self, file_path: str):
        with open(file_path, mode="r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            teams = list(reader.fieldnames)[2:]  # 获取所有球队简称

            for row in reader:
                home_team_abbr = self.teamabbr_index.get_team_abbr_with_name(row["Team"])
                for away_team_abbr in teams:
                    if away_team_abbr != home_team_abbr and row[away_team_abbr]:
                        score_str = row[away_team_abbr]
                        if "-" in score_str:
                            home_score, away_score = map(int, score_str.split("-"))
                            self.add_history(
                                home_team_abbr,
                                away_team_abbr,
                                home_score - away_score,
                            )
