import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import time
import unicodedata

st.set_page_config(page_title="Best XI Predictor", layout="wide")
st.title("⚽ Best XI Selector - Premier League Clash")

logo_dir = "club_logos"
team_files = sorted([f.replace(".csv", "") for f in os.listdir() if f.endswith(".csv")])
display_names = {team: team.replace("_", " ") for team in team_files}
reverse_names = {v: k for k, v in display_names.items()}

col1, col2 = st.columns(2)
with col1:
    team1_display = st.selectbox("Select Team 1", list(display_names.values()), index=0)
with col2:
    team2_display = st.selectbox("Select Team 2", list(display_names.values()), index=1)

team1 = reverse_names[team1_display]
team2 = reverse_names[team2_display]

logo1_path = os.path.join(logo_dir, f"{team1}.png")
logo2_path = os.path.join(logo_dir, f"{team2}.png")

cols = st.columns(2)
with cols[0]:
    if os.path.exists(logo1_path):
        st.image(logo1_path, width=150)
with cols[1]:
    if os.path.exists(logo2_path):
        st.image(logo2_path, width=150)

if team1 == team2:
    st.warning("Please select two different teams.")
    st.stop()

def load_and_clean_csv(file):
    df = pd.read_csv(file, encoding='latin1')  # Handles weird Unicode
    df.columns = [col.strip().replace('\ufeff', '') for col in df.columns]
    df.columns = [unicodedata.normalize("NFKD", str(col)) for col in df.columns]
    df[df.columns[0]] = df[df.columns[0]].apply(lambda x: unicodedata.normalize("NFKD", str(x)))
    return df

df1 = load_and_clean_csv(f"{team1}.csv")
df2 = load_and_clean_csv(f"{team2}.csv")

vs_column1 = f"vs_{team2.replace(' ', '_')}"
vs_column2 = f"vs_{team1.replace(' ', '_')}"

df1 = df1[(df1["injured"] == 0) & (df1["suspended"] == 0)]
df2 = df2[(df2["injured"] == 0) & (df2["suspended"] == 0)]

st.subheader("Formation Selection")
formation_options = {
    "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
    "4-2-3-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
    "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
    "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
    "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2}
}

formation_cols = st.columns(2)
with formation_cols[0]:
    formation1_str = st.selectbox(f"{team1_display} Formation", list(formation_options.keys()), index=0)
with formation_cols[1]:
    formation2_str = st.selectbox(f"{team2_display} Formation", list(formation_options.keys()), index=0)

formation1 = formation_options[formation1_str]
formation2 = formation_options[formation2_str]

def select_best_11(df, opponent_vs_column, formation):
    best_11 = []
    for pos, count in formation.items():
        position_df = df[df["position"] == pos]
        if len(position_df) == 0:
            continue
        if len(position_df) <= count:
            best_11.append(position_df)
        else:
            features = ["form", opponent_vs_column, "recent_starts"]
            scaler = StandardScaler()
            position_df_scaled = scaler.fit_transform(position_df[features])
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            position_df["starter"] = kmeans.fit_predict(position_df_scaled)
            starter_cluster = position_df.groupby("starter")["form"].mean().idxmax()
            best_in_cluster = position_df[position_df["starter"] == starter_cluster]
            if len(best_in_cluster) < count:
                best_11.append(position_df.nlargest(count, "form"))
            else:
                best_11.append(best_in_cluster.nlargest(count, "form"))
    best_11 = pd.concat(best_11)
    if len(best_11) < 11:
        remaining = df[~df[df.columns[0]].isin(best_11[best_11.columns[0]])]
        best_11 = pd.concat([best_11, remaining.nlargest(11 - len(best_11), "form")])
    best_11 = best_11.rename(columns={best_11.columns[0]: "player_name"})
    best_11 = best_11.sort_values(by=["position"], key=lambda x: x.map({"GK": 1, "DEF": 2, "MID": 3, "FWD": 4}))
    return best_11

if st.button("Generate Starting XIs"):
    team1_xi = select_best_11(df1, vs_column1, formation1)
    team2_xi = select_best_11(df2, vs_column2, formation2)

    st.subheader("📝 Starting Lineups Revealed by Position")

    position_map = {
        "GK": "🧤 Goalkeeper",
        "DEF": "🛡️ Defenders",
        "MID": "🎯 Midfielders",
        "FWD": "⚡ Forwards"
    }

    for pos in ["GK", "DEF", "MID", "FWD"]:
        team1_pos = team1_xi[team1_xi["position"] == pos]
        team2_pos = team2_xi[team2_xi["position"] == pos]

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**🔵 {team1_display} - {position_map[pos]}**")
            for _, row in team1_pos.iterrows():
                st.markdown(f"- {row['player_name']}")
        with col2:
            st.markdown(f"**🔴 {team2_display} - {position_map[pos]}**")
            for _, row in team2_pos.iterrows():
                st.markdown(f"- {row['player_name']}")
        
        time.sleep(1.2)
