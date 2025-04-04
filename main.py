import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from src.models import *
import math


def train(data, label, num_round: int = 100, test_data_save=False):
    train_data = lgb.Dataset(data, label=label)
    if test_data_save:
        train_data.save_binary("train.bin")
    bst = lgb.train(
        params={"learning_rate": 0.1},
        train_set=train_data,
        num_boost_round=num_round,
    )
    return bst


def get_15top_players_in_team(team):
    sorted_players = sorted(team.players, key=lambda x: x.pts, reverse=True)
    top_15_players = sorted_players[:15]
    top_15_players_feature = [player.get_features()[1] for player in top_15_players]
    top_15_players_feature_sum = np.empty(len(top_15_players_feature[0]))
    for row in top_15_players_feature:
        for i, attr in enumerate(row):
            top_15_players_feature_sum[i] += attr
    return top_15_players_feature_sum


def preprocess():
    global allteam
    allteam = AllTeam()
    global teamvsteam
    teamvsteam = TeamVSTeam()
    teamvsteam.read_and_process_csv("./data/team_vs_team.csv")
    global players
    players = Player.import_from_csv(Player, "./data/player_stats_total.csv")
    ignore_team = ["2TM", "3TM"]
    # global pos
    # pos = []
    for player in players:
        if player.team in ignore_team:
            continue
        if type(player.team) != str:
            continue
        # if player.pos not in pos:
        #     pos.append(player.pos)
        allteam.get_team_with_name(player.team).add_player(player)
    num_teams = len(allteam.get_all_teams())
    global num_feature
    num_feature = len(players[0].get_features()[1])
    global result_feature
    result_feature = np.empty((num_teams, num_feature))  # 预分配空间
    for i, team in enumerate(allteam.get_all_teams()):
        result_feature[i] = get_15top_players_in_team(team)
        # result_feature correspond with the order of teams in allteam.get_all_teams()
    global final_feature
    final_feature = np.empty(num_feature)
    global final_label
    final_label = np.empty(1)
    global feature_index
    feature_index = {}
    for i, host_team in enumerate(result_feature):
        for j, guest_team in enumerate(result_feature):
            if i != j:
                host_team_name_abbr = allteam.get_all_teams()[i].name_abbr
                guest_team_name_abbr = allteam.get_all_teams()[j].name_abbr
                # print(
                #     host_team_name_abbr,
                #     "vs",
                #     guest_team_name_abbr,
                #     ":",
                #     teamvsteam.get_history(host_team_name_abbr, guest_team_name_abbr)
                # )
                final_feature = np.vstack((final_feature, host_team - guest_team))
                final_label = np.append(
                    final_label,
                    teamvsteam.get_history(host_team_name_abbr, guest_team_name_abbr),
                )
                feature_index.update(
                    {
                        (host_team_name_abbr, guest_team_name_abbr): host_team
                        - guest_team
                    }
                )
    final_label = final_label[1:]
    final_feature = final_feature[1:]
    return (final_feature, final_label)


def get_feature_when_team_vs_team(teamname1: str, teamname2: str):
    return feature_index.get(
        (
            allteam.get_team_abbr_with_name(teamname1),
            allteam.get_team_abbr_with_name(teamname2),
        )
    )


def main():
    feature, label = preprocess()
    global bst
    bst = train(feature, label, num_round=10)
    global pred_feature
    pred_feature = np.empty(num_feature)
    pred_feature = np.vstack(
        (
            pred_feature,
            get_feature_when_team_vs_team("Milwaukee Bucks", "New Orleans Pelicans"),
        )
    )
    pred_feature = np.vstack(
        (
            pred_feature,
            get_feature_when_team_vs_team("New Orleans Pelicans", "Milwaukee Bucks"),
        )
    )
    pred_feature = np.vstack(
        (
            pred_feature,
            get_feature_when_team_vs_team("Houston Rockets", "Golden State Warriors"),
        )
    )
    pred_feature = np.vstack(
        (
            pred_feature,
            get_feature_when_team_vs_team("Indiana Pacers", "Denver Nuggets"),
        )
    )
    global pred
    pred = bst.predict(pred_feature[1:])


if __name__ == "__main__":
    main()
