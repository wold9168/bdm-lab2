import pandas as pd
import numpy as np

teamdict = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Chicago Bulls": "CHI",
    "Charlotte Hornets": "CHO",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# =====================================================================
# 核心数据处理逻辑
# =====================================================================
def process_teamvs_data(df):
    # 转换队伍全称为简称
    df["Team"] = df["Team"].map(teamdict)

    # 转换所有比分列为分差
    for col in df.columns[2:]:  # 从第三列开始是队伍简称
        df[col] = df[col].apply(
            lambda x: (
                (int(x.split("-")[0]) - (int(x.split("-")[1])))
                if isinstance(x, str)
                else 0
            )
        )

    # 设置索引为队伍简称
    df.set_index("Team", inplace=True)
    df.drop(columns=["Rk"], inplace=True)
    return df


# =====================================================================
# 查询接口
# =====================================================================
class TeamQuery:
    def __init__(self, df):
        self.df = df

    def get_diff(self, team1, team2):
        """获取两队间的分差（支持任意顺序查询）"""
        if team1 == team2:
            return 0
        try:
            return self.df.loc[team1, team2]
        except KeyError:
            raise ValueError("队伍简称不存在")


def teamvsteam_history(self, team_vs_team_csv="./data/team_ve_team.csv"):
    pass
