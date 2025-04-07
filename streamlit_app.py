import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson
import json
import datetime
import os

# Simulierte Spielerstatistiken (Demo)
spieler_daten = {
    "Liverpool": {
        "Salah": {"goals": 18, "assists": 8, "shots_on_target": 55, "cards": 3},
        "Nunez": {"goals": 13, "assists": 5, "shots_on_target": 42, "cards": 4}
    },
    "Man City": {
        "Haaland": {"goals": 27, "assists": 4, "shots_on_target": 60, "cards": 2},
        "De Bruyne": {"goals": 7, "assists": 17, "shots_on_target": 25, "cards": 1}
    }
}

# Funktion, um Teamdaten inkl. H2H & Heim/Auswärts zu laden
def get_team_stats(team_name, home=True):
    demo_data = {
        "Liverpool": {
            "home": {"xg": [2.1, 2.3, 2.0], "shots": [16, 18, 17], "shots_on_target": [7, 8, 6],
                      "corners": [8, 9, 8], "cards": [1, 2, 1], "btts": [1, 1, 0], "over_2_5": [1, 1, 1]},
            "away": {"xg": [1.5, 1.8, 1.6], "shots": [13, 15, 14], "shots_on_target": [5, 6, 5],
                      "corners": [6, 7, 6], "cards": [1, 1, 2], "btts": [0, 1, 1], "over_2_5": [1, 0, 1]},
            "h2h": {"vs Man City": [(2, 2), (1, 1), (1, 3)]}
        },
        "Man City": {
            "home": {"xg": [2.4, 2.2, 2.3], "shots": [18, 19, 17], "shots_on_target": [8, 9, 7],
                      "corners": [9, 8, 10], "cards": [1, 1, 1], "btts": [1, 1, 1], "over_2_5": [1, 1, 1]},
            "away": {"xg": [2.0, 1.9, 1.8], "shots": [16, 15, 14], "shots_on_target": [7, 6, 5],
                      "corners": [8, 7, 8], "cards": [1, 2, 1], "btts": [1, 0, 1], "over_2_5": [1, 1, 0]},
            "h2h": {"vs Liverpool": [(2, 2), (1, 1), (3, 1)]}
        }
    }

    if team_name not in demo_data:
        return None, "", ""

    side = "home" if home else "away"
    stats = demo_data[team_name][side]
    h2h = demo_data[team_name].get("h2h", {})
    players = spieler_daten.get(team_name, {})
    return stats, h2h, players

def aggregate_stats(stats):
    return {
        "avg_shots": round(np.mean(stats["shots"]), 1),
        "avg_shots_on_target": round(np.mean(stats["shots_on_target"]), 1),
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

def predict_outcome(avg_home_xg, avg_away_xg):
    if avg_home_xg > avg_away_xg + 0.3:
        return "1 (Heimsieg)"
    elif avg_away_xg > avg_home_xg + 0.3:
        return "2 (Auswärtssieg)"
    else:
        return "X (Unentschieden)"

def fetch_actual_result(home_team, away_team):
    return {"home_goals": 2, "away_goals": 1}  # Platzhalter

def evaluate_prediction(predicted_scores, actual_result):
    top_pred = predicted_scores[0][0]
    return top_pred == (actual_result["home_goals"], actual_result["away_goals"])

def save_feedback(home_team, away_team, predicted, actual, correct):
    data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "home_team": home_team,
        "away_team": away_team,
        "predicted": predicted,
        "actual": actual,
        "correct": correct
    }
    if not os.path.exists("feedback.json"):
        with open("feedback.json", "w") as f:
            json.dump([data], f, indent=4)
    else:
        with open("feedback.json", "r") as f:
            existing = json.load(f)
        existing.append(data)
        with open("feedback.json", "w") as f:
            json.dump(existing, f, indent=4)

# Streamlit UI
st.title("⚽️ Fußball-Prognose Tool mit Lernfunktion & Spielerstatistiken")

home_team = st.text_input("Heimteam", "Liverpool")
away_team = st.text_input("Auswärtsteam", "Man City")

if st.button("🔍 Prognose starten"):
    with st.spinner("Lade Daten inkl. Spieler & Lernlogik..."):
        home_stats_raw, home_h2h, home_players = get_team_stats(home_team, home=True)
        away_stats_raw, away_h2h, away_players = get_team_stats(away_team, home=False)

        if home_stats_raw is None or away_stats_raw is None:
            st.error("❌ Teamdaten nicht gefunden.")
        else:
            home_stats = aggregate_stats(home_stats_raw)
            away_stats = aggregate_stats(away_stats_raw)

            st.subheader(f"📊 Statistik-Vergleich: {home_team} vs. {away_team}")
            st.write(f"- Schüsse gesamt: {home_stats['avg_shots']} vs {away_stats['avg_shots']}")
            st.write(f"- Schüsse aufs Tor: {home_stats['avg_shots_on_target']} vs {away_stats['avg_shots_on_target']}")
            st.write(f"- Ecken: {home_stats['avg_corners']} vs {away_stats['avg_corners']}")
            st.write(f"- Karten: {home_stats['avg_cards']} vs {away_stats['avg_cards']}")
            st.write(f"- BTTS: {round((home_stats['btts_prob'] + away_stats['btts_prob']) / 2)}%")
            st.write(f"- Over 2.5: {round((home_stats['over_25_prob'] + away_stats['over_25_prob']) / 2)}%")
            st.write(f"- xG: {home_stats['avg_xg']} vs {away_stats['avg_xg']}")

            outcome = predict_outcome(home_stats['avg_xg'], away_stats['avg_xg'])
            st.subheader(f"🔮 1X2-Prognose: {outcome}")

            predicted_scores = predict_scores(home_stats['avg_xg'], away_stats['avg_xg'])
            st.subheader("🎯 Wahrscheinlichste Ergebnisse:")
            for (score, prob) in predicted_scores:
                st.write(f"- {score[0]}:{score[1]} ({prob}%)")

            if home_h2h.get(f"vs {away_team}"):
                st.subheader("📚 H2H-Duelle:")
                for result in home_h2h[f"vs {away_team}"]:
                    st.write(f"- {home_team} {result[0]}:{result[1]} {away_team}")

            st.subheader("👤 Spielerstatistiken")
            for team, players in zip([home_team, away_team], [home_players, away_players]):
                st.markdown(f"**{team}**")
                for name, stats in players.items():
                    st.write(f"{name}: Tore {stats['goals']}, Assists {stats['assists']}, SOT {stats['shots_on_target']}, Karten {stats['cards']}")

            actual_result = fetch_actual_result(home_team, away_team)
            korrekt = evaluate_prediction(predicted_scores, actual_result)
            save_feedback(home_team, away_team, predicted_scores[0][0], (actual_result['home_goals'], actual_result['away_goals']), korrekt)

            st.subheader("📈 Lern-Modul")
            st.write(f"Echtes Ergebnis: {actual_result['home_goals']}:{actual_result['away_goals']}")
            st.success("✅ Prognose korrekt!") if korrekt else st.error("❌ Prognose falsch – wird gespeichert für Lernsystem")
