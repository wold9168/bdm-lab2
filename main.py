import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error  # 新增回归指标
from sklearn.model_selection import train_test_split, cross_val_score
from skopt import BayesSearchCV
from src.models import *
import math
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor

def train(data, label, params: dict, num_round: int = 100, test_data_save=False):
    model = LGBMRegressor(n_estimators=num_round, **params)
    model.fit(data, label)
    return model


# 新增超参数搜索函数
def hyperparameter_tuning(X, y):
    search_space = {
        "learning_rate": (0.01, 0.3, "log-uniform"),
        "num_leaves": (20, 100),
        "max_depth": (3, 15),
        "min_child_samples": (10, 100),
        "subsample": (0.7, 1.0, "uniform"),
        "colsample_bytree": (0.7, 1.0, "uniform"),
    }

    # 修改为回归设置
    opt = BayesSearchCV(
        estimator=LGBMRegressor(verbose=-1),
        search_spaces=search_space,
        n_iter=30,
        cv=3,
        scoring="neg_mean_squared_error",  # 使用负均方误差作为评估指标
        n_jobs=-1,
    )
    opt.fit(X, y)

    print("最佳参数:", opt.best_params_)
    print("最佳MSE:", -opt.best_score_)  # 注意取负号得到真实MSE
    return opt.best_estimator_


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

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        feature, label, test_size=0.2, random_state=42
    )

    # 超参数调优
    print("开始超参数调优...")
    best_model = hyperparameter_tuning(X_train, y_train)

    # 最终模型
    global bst
    bst = best_model

    # 回归评估
    predicted = bst.predict(X_test)
    print(f"测试集MSE: {mean_squared_error(y_test, predicted):.2f}")
    print(f"测试集MAE: {mean_absolute_error(y_test, predicted):.2f}")

    # 原有预测逻辑保持不变
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
    predicted = bst.predict(pred_feature[1:])
    print(predicted)


if __name__ == "__main__":
    main()
