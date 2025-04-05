import pandas as pd
import math

class Player:
    def __init__(
        self,
        rank,
        player_name,
        age,
        team,
        pos,
        g,
        gs,
        mp,
        fg,
        fga,
        fg_pct,
        three_p,
        three_pa,
        three_p_pct,
        two_p,
        two_pa,
        two_p_pct,
        efg_pct,
        ft,
        fta,
        ft_pct,
        orb,
        drb,
        trb,
        ast,
        stl,
        blk,
        tov,
        pf,
        pts,
        trip_dbl,
        awards,
        player_additional,
    ):
        self.rank = rank
        self.player_name = player_name
        self.age = age
        self.team = team
        self.pos = pos
        self.g = g
        self.gs = gs
        self.mp = mp
        self.fg = fg
        self.fga = fga
        self.fg_pct = fg_pct
        self.three_p = three_p
        self.three_pa = three_pa
        self.three_p_pct = three_p_pct
        self.two_p = two_p
        self.two_pa = two_pa
        self.two_p_pct = two_p_pct
        self.efg_pct = efg_pct
        self.ft = ft
        self.fta = fta
        self.ft_pct = ft_pct
        self.orb = orb
        self.drb = drb
        self.trb = trb
        self.ast = ast
        self.stl = stl
        self.blk = blk
        self.tov = tov
        self.pf = pf
        self.pts = pts
        self.trip_dbl = trip_dbl
        self.awards = awards
        self.player_additional = player_additional

    @staticmethod
    def import_from_csv(cls,csv_file_path:str):
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file_path)
        players = []
        for index, row in df.iterrows():
            player = cls(
                rank=row["Rk"],
                player_name=row["Player"],
                age=row["Age"],
                team=row["Team"],
                pos=row["Pos"],
                g=row["G"],
                gs=row["GS"],
                mp=row["MP"],
                fg=row["FG"],
                fga=row["FGA"],
                fg_pct=row["FG%"],
                three_p=row["3P"],
                three_pa=row["3PA"],
                three_p_pct=row["3P%"],
                two_p=row["2P"],
                two_pa=row["2PA"],
                two_p_pct=row["2P%"],
                efg_pct=row["eFG%"],
                ft=row["FT"],
                fta=row["FTA"],
                ft_pct=row["FT%"],
                orb=row["ORB"],
                drb=row["DRB"],
                trb=row["TRB"],
                ast=row["AST"],
                stl=row["STL"],
                blk=row["BLK"],
                tov=row["TOV"],
                pf=row["PF"],
                pts=row["PTS"],
                trip_dbl=row["Trp-Dbl"],
                awards=row["Awards"],
                player_additional=row["Player-additional"],
            )
            players.append(player)

        return players

    def __repr__(self):
        return (
            f"Player(rank={self.rank}, player_name='{self.player_name}', age={self.age}, "
            f"team='{self.team}', pos='{self.pos}', g={self.g}, gs={self.gs}, mp={self.mp}, "
            f"fg={self.fg}, fga={self.fga}, fg_pct={self.fg_pct}, three_p={self.three_p}, "
            f"three_pa={self.three_pa}, three_p_pct={self.three_p_pct}, two_p={self.two_p}, "
            f"two_pa={self.two_pa}, two_p_pct={self.two_p_pct}, efg_pct={self.efg_pct}, "
            f"ft={self.ft}, fta={self.fta}, ft_pct={self.ft_pct}, orb={self.orb}, "
            f"drb={self.drb}, trb={self.trb}, ast={self.ast}, stl={self.stl}, "
            f"blk={self.blk}, tov={self.tov}, pf={self.pf}, pts={self.pts}, "
            f"trip_dbl={self.trip_dbl}, awards='{self.awards}', "
            f"player_additional='{self.player_additional}')"
        )

    @staticmethod
    def get_features_names():
        ignore_attributes = [
            "rk",
            "player_name",
            "team",
            "pos",
            "awards",
            "player_additional",
        ]
        features_names = [
            attr
            for attr in Player.__static_attributes__
            if attr not in ignore_attributes
        ]
        return features_names

    def get_features(self):
        ignore_attributes = [
            "rk",
            "player_name",
            "team",
            "pos",
            "awards",
            "player_additional",
        ]
        features_names = [
            attr
            for attr in Player.__static_attributes__
            if attr not in ignore_attributes
        ]
        features = [self.__dict__[features_name] for features_name in features_names]
        for i, feature in enumerate(features):
            if math.isnan(feature):
                features[i] = 0.0
        return ((self.team, self.pos), features)
