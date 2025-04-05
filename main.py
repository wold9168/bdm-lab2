import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocess import *
from src.teamvs_histroy import *
from joblib import parallel_backend


def build_dataset(team_power, query):
    """将队伍战斗力数据转换为模型可用的特征矩阵"""
    # 获取所有队伍简称
    teams = team_power["Team"].tolist()

    # 生成所有可能的队伍对战组合（排除相同队伍）
    ignore_team = ["2TM", "3TM"]
    matchups = [
        (t1, t2)
        for t1 in teams
        for t2 in teams
        if t1 != t2 and t1 not in ignore_team and t2 not in ignore_team
    ]

    # 构造特征矩阵
    features = []
    labels = []

    for t1, t2 in matchups:
        # 获取两队特征（假设team_power的索引是队伍简称）
        ft1 = team_power.loc[team_power["Team"] == t1].values[0][1:]
        ft2 = team_power.loc[team_power["Team"] == t2].values[0][1:]

        # 构造特征：队伍1特征 - 队伍2特征（差值特征）
        feature_diff = ft1 - ft2
        features.append(feature_diff)

        # 获取标签：历史分差
        labels.append(query.get_diff(t1, t2))

    return np.array(features), np.array(labels)


# 评估指标
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")


def predict(model, team_power, team1: str, team2: str):
    y_pred = model.predict(
        np.array(
            team_power.loc[team_power["Team"] == team1].values[0][1:]
            - team_power.loc[team_power["Team"] == team2].values[0][1:]
        ).reshape(1, -1)
    )
    return y_pred


def main():
    # --------------------------------------------------
    # 加载数据
    # --------------------------------------------------
    # 假设预处理后的战斗力数据
    global team_power
    team_power = Preprocess().preprocess()

    # 添加BMI相关特征
    # team_power["bmi_avg"] = team_power["player_bmi"].mean()  # 平均BMI
    # team_power["bmi_diff"] = team_power["player_bmi"] - team_power["bmi_avg"]  # BMI差值

    # 加载历史对战数据
    global query
    query = TeamQuery(process_teamvs_data(pd.read_csv("./data/team_vs_team.csv")))

    # 构建数据集
    X, y = build_dataset(team_power, query)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=40
    )

    # --------------------------------------------------
    # 训练基准模型（包含BMI特征）
    # --------------------------------------------------
    # 初始参数
    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        "verbose": -1,
    }

    # 训练模型
    global model_with_bmi
    model_with_bmi = lgb.LGBMRegressor(**params)
    model_with_bmi.fit(X_train, y_train)

    print("包含BMI特征的模型表现：")
    evaluate(model_with_bmi, X_test, y_test)

    # --------------------------------------------------
    # BMI特征重要性分析
    # --------------------------------------------------
    # 获取特征名称（假设team_power的特征顺序与构造时一致）
    feature_names = [f"{col}_diff" for col in team_power.columns]

    # 可视化特征重要性
    # lgb.plot_importance(
    #     model_with_bmi, figsize=(12, 6), title="Feature Importance (with BMI)"
    # )
    # plt.show()

    # 对比实验：移除BMI特征
    X_train_no_bmi = np.delete(
        X_train,
        [
            feature_names.index("player_bmi_diff") - 1,
            # feature_names.index("bmi_avg_diff") - 1,
            # feature_names.index("bmi_diff_diff") - 1,
        ],
        axis=1,
    )
    X_test_no_bmi = np.delete(
        X_test,
        [
            feature_names.index("player_bmi_diff") - 1,
            # feature_names.index("bmi_avg_diff") - 1,
            # feature_names.index("bmi_diff_diff") - 1,
        ],
        axis=1,
    )
    global model_without_bmi
    model_without_bmi = lgb.LGBMRegressor(**params)
    model_without_bmi.fit(X_train_no_bmi, y_train)

    print("\n不包含BMI特征的模型表现：")
    evaluate(model_without_bmi, X_test_no_bmi, y_test)

    # --------------------------------------------------
    # 超参数网格搜索
    # --------------------------------------------------
    # 定义参数网格
    param_grid = {
        "num_leaves": [10, 15, 30],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [1, 3, 5],
        "min_child_samples": [10, 30],
        "verbose": [-1],
    }
    # 网格搜索
    grid = GridSearchCV(
        estimator=lgb.LGBMRegressor(objective="regression", metric="mae"),
        param_grid=param_grid,
        cv=3,
        scoring="neg_mean_absolute_error",
        verbose=2,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    # 输出最佳参数
    print(f"最佳参数组合：{grid.best_params_}")
    print(f"最佳MAE：{-grid.best_score_:.2f}")

    # --------------------------------------------------
    # 优化后模型评估
    # --------------------------------------------------
    optimized_model = grid.best_estimator_
    print("\n优化后模型表现：")
    evaluate(optimized_model, X_test, y_test)

    # 对比优化前后
    y_pred_base = model_with_bmi.predict(X_test)
    y_pred_opt = optimized_model.predict(X_test)

    print("\n优化前后对比：")
    print(f"基准模型 MAE: {mean_absolute_error(y_test, y_pred_base):.2f}")
    print(f"优化模型 MAE: {mean_absolute_error(y_test, y_pred_opt):.2f}")
    print(
        f"提升幅度: {(mean_absolute_error(y_test, y_pred_base) - mean_absolute_error(y_test, y_pred_opt)):.2f}"
    )

    # 残差分析
    residuals = y_test - y_pred_opt
    sns.histplot(residuals, kde=True)
    plt.title("Optimized Model Residual Distribution")
    plt.xlabel("Prediction Error")
    plt.show()
    # --------------------------------------------------
    # 生成分析报告
    # --------------------------------------------------
    # BMI特征重要性数据
    bmi_importance = model_with_bmi.feature_importances_[
        feature_names.index("player_bmi_diff")
        - 1
        # Team_power(feature_names)的[0]是Team名，但是传入的训练参数是[1:]
        # In [16]: feature_names.index("player_bmi_diff")
        # Out[16]: 43
        # In [17]: team_power.columns.to_list()[1:].index("player_bmi")
        # Out[17]: 42
    ]
    avg_importance = np.mean(model_with_bmi.feature_importances_)

    # 性能对比数据
    mae_with = mean_absolute_error(y_test, model_with_bmi.predict(X_test))
    mae_without = mean_absolute_error(y_test, model_without_bmi.predict(X_test_no_bmi))

    # 参数优化提升
    mae_improvement = mean_absolute_error(y_test, y_pred_base) - mean_absolute_error(
        y_test, y_pred_opt
    )

    print(
        f"""
    ============ 分析报告 ============
    1. BMI特征影响：
        - BMI特征重要性得分：{bmi_importance}（平均特征得分：{avg_importance:.1f}）
        - 包含BMI的模型MAE：{mae_with:.2f}
        - 不含BMI的模型MAE：{mae_without:.2f}
        - BMI带来的MAE提升：{mae_without - mae_with:.2f}

    2. 超参数优化效果：
        - 优化后MAE提升：{mae_improvement:.2f}
        - 最佳参数组合：{grid.best_params_}

    3. 残差分析：
        - 残差均值：{np.mean(residuals):.2f}
        - 残差标准差：{np.std(residuals):.2f}
        - 95%预测误差范围：[{np.percentile(residuals, 2.5):.1f}, {np.percentile(residuals, 97.5):.1f}]
    ================================
    """
    )


if __name__ == "__main__":
    main()
