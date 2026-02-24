import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import os

st.set_page_config(page_title="T20 Predictor", layout="wide")

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
   pd.read_csv("ICC Mens T20 Worldcup.csv")
    return df

df = load_data()

st.title("🏏 T20 World Cup AI Predictor")

# ===============================
# DATA PREPARATION
# ===============================
data = df[['1st Team', '2nd Team', 'Winners']].dropna()

le = LabelEncoder()
all_teams = pd.concat([data['1st Team'], data['2nd Team'], data['Winners']])
le.fit(all_teams)

data['Team1'] = le.transform(data['1st Team'])
data['Team2'] = le.transform(data['2nd Team'])
data['Winner'] = le.transform(data['Winners'])

X = data[['Team1', 'Team2']]
y = data['Winner']

# Train Model
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X, y)

# ===============================
# SIDEBAR MENU
# ===============================
menu = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Team Analysis",
    "Match Prediction",
    "Predict 2026 Champion"
])

# ===============================
# DASHBOARD
# ===============================
if menu == "Dashboard":

    st.subheader("📊 Dataset Overview")
    st.write(df.head())

    st.subheader("Total Matches Played")
    st.metric("Matches", len(df))

# ===============================
# TEAM ANALYSIS
# ===============================
elif menu == "Team Analysis":

    st.subheader("🏆 Most Successful Teams")

    win_counts = df['Winners'].value_counts()

    fig, ax = plt.subplots()
    win_counts.plot(kind='bar', ax=ax)
    ax.set_title("Team Wins")
    ax.set_ylabel("Wins")

    st.pyplot(fig)

    st.subheader("📈 Team Strength Ranking")

    matches = pd.concat([df['1st Team'], df['2nd Team']]).value_counts()
    wins = df['Winners'].value_counts()

    strength = (wins / matches) * 100
    strength = strength.sort_values(ascending=False)

    st.dataframe(strength)

# ===============================
# MATCH PREDICTION
# ===============================
elif menu == "Match Prediction":

    st.subheader("⚡ Predict Match Winner")

    teams = sorted(le.classes_)

    team1 = st.selectbox("Select Team 1", teams)
    team2 = st.selectbox("Select Team 2", teams)

    if st.button("Predict Winner"):

        t1 = le.transform([team1])[0]
        t2 = le.transform([team2])[0]

        pred = model.predict([[t1, t2]])
        prob = model.predict_proba([[t1, t2]])

        winner = le.inverse_transform(pred)[0]
        probability = max(prob[0]) * 100

        st.success(f"🏆 Predicted Winner: {winner}")
        st.info(f"Winning Probability: {round(probability,2)} %")

# ===============================
# PREDICT 2026 CHAMPION
# ===============================
elif menu == "Predict 2026 Champion":

    st.subheader("🔮 2026 World Cup Champion Prediction")

    matches = pd.concat([df['1st Team'], df['2nd Team']]).value_counts()
    wins = df['Winners'].value_counts()

    strength = (wins / matches) * 100
    strength = strength.sort_values(ascending=False)

    champion = strength.index[0]

    st.success(f"🏆 Predicted Champion: {champion}")

    st.bar_chart(strength.head(10))
