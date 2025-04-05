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


def hyperparameter_tuning(X, y):
    search_space = {
        "learning_rate": (0.01, 0.3, "log-uniform"),  # 学习率范围
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
    initialize_globals()
    load_team_vs_team_data("./data/team_vs_team.csv")
    players = load_players("./data/player_stats_total.csv")
    filtered_players = filter_players(players)
    assign_players_to_teams(filtered_players)
    global team_features
    team_features = compute_all_team_features()
    # 加新特征的话加到team_features
    pair_features, pair_labels = build_dataset(team_features)
    return pair_features, pair_labels


def initialize_globals():
    global allteam, teamvsteam, feature_index
    allteam = AllTeam()
    teamvsteam = TeamVSTeam()
    feature_index = {}


def load_team_vs_team_data(filepath):
    teamvsteam.read_and_process_csv(filepath)


def load_players(filepath):
    return Player.import_from_csv(Player, filepath)


def filter_players(raw_players, ignore_teams=["2TM", "3TM"]):
    return [
        p for p in raw_players if isinstance(p.team, str) and p.team not in ignore_teams
    ]


def assign_players_to_teams(valid_players):
    for player in valid_players:
        allteam.get_team_with_name(player.team).add_player(player)


def compute_team_features(team):
    return get_15top_players_in_team(team)


def compute_all_team_features():
    teams = allteam.get_all_teams()
    global num_features
    num_features = len(teams[0].players[0].get_features()[1])
    features = np.empty((len(teams), num_features))

    for i, team in enumerate(teams):
        features[i] = compute_team_features(team)
    return features


def build_dataset(team_features):
    teams = allteam.get_all_teams()
    feature_diffs = []
    labels = []

    for i, host_feature in enumerate(team_features):
        for j, guest_feature in enumerate(team_features):
            if i == j:
                continue

            host_abbr = teams[i].name_abbr
            guest_abbr = teams[j].name_abbr

            # 计算特征差异
            feature_diff = host_feature - guest_feature
            feature_diffs.append(feature_diff)

            # 获取历史记录
            history = teamvsteam.get_history(host_abbr, guest_abbr)
            labels.append(history)

            # 记录特征索引
            feature_index[(host_abbr, guest_abbr)] = feature_diff

    return np.array(feature_diffs), np.array(labels)


def get_feature_when_team_vs_team(teamname1: str, teamname2: str):
    return feature_index.get(
        (
            allteam.get_team_abbr_with_name(teamname1),
            allteam.get_team_abbr_with_name(teamname2),
        )
    )


def main():
    feature, label = preprocess()
    df_feature = pd.DataFrame(feature, columns=Player.get_features_names())
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        df_feature, label, test_size=0.2, random_state=42
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
    pred_feature = np.array(
        get_feature_when_team_vs_team("Milwaukee Bucks", "New Orleans Pelicans"),
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

    predicted = bst.predict(pred_feature)
    print(predicted)
    rawmodel = train(feature, label, 5)
    raw_predicted = rawmodel.predict(pred_feature)
    # print(f"原始模型的测试集MSE: {mean_squared_error(y_test, raw_predicted):.2f}")
    # print(f"原始模型的测试集MAE: {mean_absolute_error(y_test, raw_predicted):.2f}")
    print(raw_predicted)


if __name__ == "__main__":
    main()
