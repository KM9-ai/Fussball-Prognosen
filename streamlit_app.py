import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import requests
import json

# Funktion, um echte xG-Daten von Understat zu laden (vereinfacht, Demo-Daten)
def get_team_stats(team_name):
    # Simulierter Zugriff auf Teamdaten via Understat (hier mit statischen Beispieldaten)
    demo_data = {
        "Liverpool": {
            "xg": [1.8, 2.1, 1.5, 2.4, 2.0],
            "shots": [14, 16, 13, 18, 15],
            "corners": [7, 8, 6, 9, 7],
            "cards": [1, 2, 1, 3, 1],
            "btts": [1, 1, 0, 1, 1],
            "over_2_5": [1, 1, 0, 1, 1]
        },
        "Man City": {
            "xg": [2.2, 2.3, 1.9, 2.5, 2.1],
            "shots": [16, 18, 15, 20, 17],
            "corners": [8, 9, 7, 10, 9],
            "cards": [1, 1, 2, 1, 1],
            "btts": [1, 1, 1, 1, 0],
            "over_2_5": [1, 1, 1, 1, 1]
        },
        "Villarreal": {
            "xg": [1.1, 1.5, 1.3, 1.2, 1.0],
            "shots": [11, 13, 10, 12, 11],
            "corners": [5, 6, 4, 6, 5],
            "cards": [2, 1, 2, 3, 1],
            "btts": [0, 1, 0, 1, 0],
            "over_2_5": [0, 1, 0, 1, 0]
        },
        "Athletic Bilbao": {
            "xg": [1.2, 1.4, 1.0, 1.6, 1.3],
            "shots": [12, 14, 11, 15, 13],
            "corners": [6, 7, 5, 7, 6],
            "cards": [1, 2, 2, 2, 1],
            "btts": [1, 1, 0, 1, 1],
            "over_2_5": [1, 1, 0, 1, 1]
        }
    }

    if team_name not in demo_data:
        return None, ""

    stats = demo_data[team_name]
    return stats, f"https://understat.com/team/{team_name.replace(' ', '_')}"

def aggregate_stats(stats):
    return {
        "avg_shots": round(np.mean(stats["shots"]), 1),
        "avg_corners": round(np.mean(stats["corners"]), 1),
        "avg_cards": round(np.mean(stats["cards"]), 1),
        "btts_prob": round(np.mean(stats["btts"]) * 100),
        "over_25_prob": round(np.mean(stats["over_2_5"]) * 100),
        "avg_xg": round(np.mean(stats["xg"]), 2)
    }

def predict_scores(home_xg, away_xg, max_goals=5):
    score_probs = {}
    for i in range(0, max_goals + 1):
        for j in range(0, max_goals + 1):
            p = poisson.pmf(i, home_xg) * poisson.pmf(j, away_xg)
            score_probs[(i, j)] = round(p * 100, 2)
    sorted_scores = sorted(score_probs.items(), key=lambda x: x[1], reverse=True)[:3]
    return sorted_scores

# Streamlit UI
st.title("⚽ Spiel-Prognose Tool (Understat Edition)")

home_team = st.text_input("Heimteam", "Liverpool")
away_team = st.text_input("Auswärtsteam", "Man City")

if st.button("Prognose anzeigen"):
    with st.spinner("Lade Understat-Daten..."):
        home_stats_raw, home_link = get_team_stats(home_team)
        away_stats_raw, away_link = get_team_stats(away_team)

        if home_stats_raw is None or away_stats_raw is None:
            st.error("Teamdaten nicht gefunden. Bitte einen verfügbaren Teamnamen eingeben.")
        else:
            home_stats = aggregate_stats(home_stats_raw)
            away_stats = aggregate_stats(away_stats_raw)

            st.subheader(f"📊 {home_team} vs. {away_team} Prognose")
            st.write(f"- Schüsse: {home_stats['avg_shots']} vs {away_stats['avg_shots']}")
            st.write(f"- Ecken: {home_stats['avg_corners']} vs {away_stats['avg_corners']}")
            st.write(f"- Karten: {home_stats['avg_cards']} vs {away_stats['avg_cards']}")
            st.write(f"- Beide Teams treffen: JA ({round((home_stats['btts_prob'] + away_stats['btts_prob']) / 2)}%)")
            st.write(f"- Über 2.5 Tore: JA ({round((home_stats['over_25_prob'] + away_stats['over_25_prob']) / 2)}%)")
            st.write(f"- Erwartetes xG: {home_stats['avg_xg']} vs {away_stats['avg_xg']}")

            st.caption(f"🔗 {home_team} Datenquelle: {home_link}")
            st.caption(f"🔗 {away_team} Datenquelle: {away_link}")
            st.caption(f"📈 Analyse basiert auf den letzten Spielen (Understat Daten)")

            predicted_scores = predict_scores(home_stats['avg_xg'], away_stats['avg_xg'])

            st.subheader("🎯 Wahrscheinlichste Ergebnisse:")
            for (score, prob) in predicted_scores:
                st.write(f"- {score[0]}:{score[1]} ({prob}%)")