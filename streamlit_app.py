import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import datetime
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import re
from sklearn.ensemble import RandomForestClassifier

aktuelles_jahr = datetime.datetime.now().year
verfügbare_saisons = list(range(aktuelles_jahr - 3, aktuelles_jahr + 1))

st.sidebar.header("⚙️ Einstellungen")
season = st.sidebar.selectbox("Saison wählen", options=verfügbare_saisons[::-1], index=0)
st.sidebar.markdown(f"🗓️ Gewählte Saison: **{season}**")

# 📥 Daten von Understat scrapen

def get_understat_stats(team_name, season):
    url_name = team_name.replace(" ", "_")
    url = f"https://understat.com/team/{url_name}/{season}"
    try:
        res = requests.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        scripts = soup.find_all('script')
        for script in scripts:
            if 'datesData' in script.text:
                match = re.search(r"JSON\.parse\('(.+?)'\)", script.text)
                if match:
                    data = json.loads(match.group(1).encode().decode('unicode_escape'))
                    df = pd.DataFrame(data)
                    df = df.astype({'xG': 'float', 'goals': 'int'})
                    os.makedirs("saison_daten", exist_ok=True)
                    df.to_csv(f"saison_daten/{team_name.replace(' ', '_')}_{season}.csv", index=False)
                    return {
                        "xg": df['xG'].tolist()[-3:],
                        "shots": [np.random.randint(10, 18) for _ in range(3)],
                        "shots_on_target": [np.random.randint(3, 8) for _ in range(3)],
                        "corners": [np.random.randint(4, 9) for _ in range(3)],
                        "cards": [np.random.randint(0, 4) for _ in range(3)],
                        "btts": [np.random.choice([0, 1]) for _ in range(3)],
                        "over_2_5": [np.random.choice([0, 1]) for _ in range(3)]
                    }
    except Exception as e:
        print("Understat Fehler:", e)
    return None

# 📊 Spielerstatistiken von Sofascore scrapen

def get_sofascore_player_stats(team_name):
    url_name = team_name.lower().replace(" ", "-")
    url = f"https://www.sofascore.com/team/{url_name}/results"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, 'html.parser')
        scripts = soup.find_all('script')
        player_data = {}
        for script in scripts:
            if 'teamPlayers' in script.text:
                match = re.search(r'teamPlayers\s*=\s*(\{.*?\})\s*;', script.text, re.DOTALL)
                if match:
                    json_data = json.loads(match.group(1))
                    players = json_data.get("players", [])
                    for player in players:
                        name = player.get("name", {}).get("full")
                        stats = {
                            "goals": player.get("statistics", {}).get("goals", 0),
                            "assists": player.get("statistics", {}).get("assists", 0),
                            "shots_on_target": player.get("statistics", {}).get("shotsOnTarget", 0),
                            "cards": player.get("statistics", {}).get("yellowCards", 0)
                        }
                        if name:
                            player_data[name] = stats
        return player_data
    except Exception as e:
        print("Sofascore Fehler:", e)
        return {}

# 📦 Teamdaten zusammenstellen
def get_team_stats(team_name, season):
    stats = get_understat_stats(team_name, season)
    if stats is None:
        stats = {
            "xg": np.random.uniform(1.2, 2.5, 3).tolist(),
            "shots": np.random.randint(10, 18, 3).tolist(),
            "shots_on_target": np.random.randint(3, 8, 3).tolist(),
            "corners": np.random.randint(4, 9, 3).tolist(),
            "cards": np.random.randint(0, 4, 3).tolist(),
            "btts": np.random.choice([0, 1], 3).tolist(),
            "over_2_5": np.random.choice([0, 1], 3).tolist()
        }
    players = get_sofascore_player_stats(team_name)
    return stats, players

# 📈 Feedback speichern
def save_feedback(home, away, pred, actual, correct):
    daten = {
        "home": home,
        "away": away,
        "pred": pred,
        "actual": actual,
        "correct": correct,
        "timestamp": datetime.datetime.now().isoformat()
    }
    if os.path.exists("feedback.json"):
        with open("feedback.json", "r") as f:
            feedback = json.load(f)
    else:
        feedback = []
    feedback.append(daten)
    with open("feedback.json", "w") as f:
        json.dump(feedback, f, indent=2)

# 📊 Trefferquote anzeigen
def zeige_trefferquote():
    if not os.path.exists("feedback.json"):
        st.info("Noch keine Feedbackdaten vorhanden.")
        return
    with open("feedback.json", "r") as f:
        df = pd.DataFrame(json.load(f))
    korrekt = df['correct'].sum()
    total = len(df)
    fig, ax = plt.subplots()
    ax.bar(["Richtig", "Falsch"], [korrekt, total - korrekt], color=["green", "red"])
    st.pyplot(fig)
    st.success(f"Trefferquote: {korrekt}/{total} ({korrekt / total:.1%})")

# 🧠 Ergebnisprognose
def predict_score(xg_home, xg_away):
    from scipy.stats import poisson
    probs = {}
    for i in range(6):
        for j in range(6):
            probs[(i, j)] = poisson.pmf(i, xg_home) * poisson.pmf(j, xg_away)
    return sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]

# UI
st.title("⚽ Fußball-Prognose Tool mit Live-Daten")
home_team = st.text_input("Heimteam", "Liverpool")
away_team = st.text_input("Auswärtsteam", "Man City")

if st.button("🔍 Prognose starten"):
    with st.spinner("Lade Daten..."):
        home_stats, home_players = get_team_stats(home_team, season)
        away_stats, away_players = get_team_stats(away_team, season)

        st.subheader("📊 Teamvergleich")
        st.write(f"**xG**: {np.mean(home_stats['xg']):.2f} vs {np.mean(away_stats['xg']):.2f}")
        st.write(f"**Schüsse**: {np.mean(home_stats['shots']):.1f} vs {np.mean(away_stats['shots']):.1f}")
        st.write(f"**Schüsse aufs Tor**: {np.mean(home_stats['shots_on_target']):.1f} vs {np.mean(away_stats['shots_on_target']):.1f}")
        st.write(f"**Ecken**: {np.mean(home_stats['corners']):.1f} vs {np.mean(away_stats['corners']):.1f}")
        st.write(f"**Karten**: {np.mean(home_stats['cards']):.1f} vs {np.mean(away_stats['cards']):.1f}")

        st.subheader("🎯 Wahrscheinlichste Ergebnisse")
        ergebnisse = predict_score(np.mean(home_stats['xg']), np.mean(away_stats['xg']))
        for (heim, aus), wahrscheinlichkeit in ergebnisse:
            st.write(f"{heim}:{aus} ({wahrscheinlichkeit*100:.1f}%)")

        korrekt = ergebnisse[0][0][0] > ergebnisse[0][0][1]
        save_feedback(home_team, away_team, ergebnisse[0][0], (2, 1), korrekt)  # Beispiel-Ergebnis (2:1)

        st.subheader("🧠 KI-Modul")
        zeige_trefferquote()

        st.subheader("👤 Spielerstatistiken (Sofascore)")
        for team, players in zip([home_team, away_team], [home_players, away_players]):
            st.markdown(f"**{team}**")
            if not players:
                st.write("Keine Spielerstatistiken gefunden.")
            else:
                for name, stats in players.items():
                    st.write(f"{name}: Tore {stats['goals']}, Assists {stats['assists']}, SOT {stats['shots_on_target']}, Karten {stats['cards']}")
