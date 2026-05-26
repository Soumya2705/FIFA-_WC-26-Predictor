# app.py  —  FIFA World Cup 2026 Match Prediction
# Updated to match notebook v3: new BOOST values, score-first knockout logic,
# static HTML serving (no Jinja template variables).

from flask import Flask, request, jsonify, send_from_directory
import joblib
import numpy as np
import pandas as pd
import os

# ── Load saved model artefacts ───────────────────────────────────────────────
xgb_model       = joblib.load('model_xgb.pkl')        # trained XGBClassifier
feature_columns  = joblib.load('feature_columns.pkl')  # ordered list of 18 feature names
team_feature_v3  = joblib.load('team_feature.pkl')     # per-team feature DataFrame
feature_scaler   = joblib.load('feature_scaler.pkl')   # fitted MinMaxScaler

# Historical matches are needed for H2H calculation
matches         = pd.read_csv('clean_matches.csv')
matches['date'] = pd.to_datetime(matches['date'])
matches_sorted  = matches.sort_values('date')

# ── Feature boost map — re-tuned to match notebook v3 ────────────────────────
# Changes vs previous version:
#   h2h_win_rate_diff       : 12.0  → 14.0  (strongest single signal)
#   avg_goals_conceded_diff : 10.0  → 11.0  (most consistent predictor)
#   collective_strength_diff:  8.0  →  9.0  (composite quality)
#   team_prestige_diff      :  7.0  →  8.0  (pedigree)
#   xfactor_diff            :  5.0  →  6.0  (tournament DNA)
#   tier_diff               :  2.0  →  2.5  (coarser tier signal)
#   elo_delta_diff          :  NEW  →  1.5  (12-month momentum)
BOOST = {
    'h2h_win_rate_diff':        14.0,
    'avg_goals_conceded_diff':  11.0,
    'collective_strength_diff':  9.0,
    'team_prestige_diff':        8.0,
    'xfactor_diff':              6.0,
    'tier_diff':                 2.5,
    'elo_delta_diff':            1.5,
    'rank_diff':                 0.05,
    'points_diff':               0.05,
}


def apply_feature_boost(df, cols, boost_map):
    df = df.copy()
    for col, mult in boost_map.items():
        if col in cols and col in df.columns:
            df[col] = df[col] * mult
    return df


# ── Core prediction function (mirrors notebook v3 logic exactly) ─────────────
def predict_match(home_team, away_team, neutral=True):
    tf = team_feature_v3.set_index('team')

    if home_team not in tf.index:
        raise ValueError(f"Team not found: {home_team!r}")
    if away_team not in tf.index:
        raise ValueError(f"Team not found: {away_team!r}")

    h = tf.loc[home_team]
    a = tf.loc[away_team]

    # Head-to-head win rate (last 10 meetings)
    mask = (
        ((matches_sorted['home_team'] == home_team) & (matches_sorted['away_team'] == away_team)) |
        ((matches_sorted['home_team'] == away_team) & (matches_sorted['away_team'] == home_team))
    )
    h2h_games = matches_sorted[mask].tail(10)
    if len(h2h_games) == 0:
        h2h = 0.0
    else:
        hw = (
            ((h2h_games['home_team'] == home_team) & (h2h_games['home_score'] > h2h_games['away_score'])).sum() +
            ((h2h_games['away_team'] == home_team) & (h2h_games['away_score'] > h2h_games['home_score'])).sum()
        )
        h2h = hw / len(h2h_games)

    # Build the single feature row
    raw_row = {
        'collective_strength_diff': float(h['collective_strength']) - float(a['collective_strength']),
        'team_prestige_diff':       float(h['team_prestige_norm'])  - float(a['team_prestige_norm']),
        'xfactor_diff':             float(h['xfactor'])             - float(a['xfactor']),
        'tier_diff':                float(h['tier_weight'])         - float(a['tier_weight']),
        'attack_diff':              float(h['attack_strength'])     - float(a['attack_strength']),
        'defense_diff':             float(h['defense_strength'])    - float(a['defense_strength']),
        'avg_goals_scored_diff':    float(h['avg_goals_scored'])    - float(a['avg_goals_scored']),
        'avg_goals_conceded_diff':  float(h['avg_goals_conceded'])  - float(a['avg_goals_conceded']),
        'recent_form_diff':         float(h['recent_form'])         - float(a['recent_form']),
        'form_streak_diff':         float(h['form_streak'])         - float(a['form_streak']),
        'elo_delta_diff':           float(h['elo_delta_12m'])       - float(a['elo_delta_12m']),
        'win_percentage_diff':      float(h['win_percentage'])      - float(a['win_percentage']),
        'goal_diff_ratio_diff':     float(h['goal_diff_ratio'])     - float(a['goal_diff_ratio']),
        'clean_sheet_diff':         float(h['clean_sheet_rate'])    - float(a['clean_sheet_rate']),
        'h2h_win_rate_diff':        h2h,
        'rank_diff':                float(a['Rank'])   - float(h['Rank']),
        'points_diff':              float(h['Points']) - float(a['Points']),
        'is_neutral':               int(neutral),
    }

    raw     = pd.DataFrame([raw_row])[feature_columns]
    norm    = pd.DataFrame(feature_scaler.transform(raw), columns=feature_columns)
    boosted = apply_feature_boost(norm, feature_columns, BOOST)
    proba   = xgb_model.predict_proba(boosted)[0]   # [away_win, draw, home_win]

    # Dixon-Coles expected goals
    cs_diff = raw_row['collective_strength_diff']
    pr_diff = raw_row['team_prestige_diff']
    xf_diff = raw_row['xfactor_diff']

    lam_h = float(h['avg_goals_scored'] * a['avg_goals_conceded']) * max(0.7, 1.0 + 0.25 * cs_diff)
    lam_a = float(a['avg_goals_scored'] * h['avg_goals_conceded']) * max(0.7, 1.0 - 0.18 * cs_diff)
    lam_h = max(0.1, lam_h + pr_diff * 0.25 + xf_diff * 0.15)
    lam_a = max(0.1, lam_a - pr_diff * 0.25 - xf_diff * 0.15)
    if neutral:
        avg   = (lam_h + lam_a) / 2
        lam_h = 0.85 * lam_h + 0.15 * avg
        lam_a = 0.85 * lam_a + 0.15 * avg

    return {
        'match':         f'{home_team} vs {away_team}',
        'home_win_prob': round(float(proba[2]) * 100, 1),
        'draw_prob':     round(float(proba[1]) * 100, 1),
        'away_win_prob': round(float(proba[0]) * 100, 1),
        'pred_score':    f'{max(0, int(round(lam_h)))} - {max(0, int(round(lam_a)))}',
        'lambda_home':   round(lam_h, 3),
        'lambda_away':   round(lam_a, 3),
    }


# ── Knockout match simulation — score-first, probability as tiebreaker ────────
# Matches notebook's simulate_knockout_match exactly:
#   1. Parse Dixon-Coles predicted score to determine winner.
#   2. Only fall back to probability when score is level (no draws in knockout).
def simulate_knockout_match(home, away):
    pred = predict_match(home, away, neutral=True)

    score_parts = pred['pred_score'].split(' - ')
    home_goals  = int(score_parts[0])
    away_goals  = int(score_parts[1])

    if home_goals > away_goals:
        winner = home
    elif away_goals > home_goals:
        winner = away
    else:
        # Drawn score — probability tiebreaker (knockout: no draws allowed)
        winner = home if pred['home_win_prob'] >= pred['away_win_prob'] else away

    loser = away if winner == home else home
    return {**pred, 'winner': winner, 'loser': loser, 'home': home, 'away': away}


# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='templates', static_url_path='')

ALL_TEAMS = sorted(team_feature_v3['team'].tolist())


# ── Serve the static index.html ───────────────────────────────────────────────
@app.route('/')
def home():
    # The new index.html is fully self-contained (no Jinja variables).
    # Flask will serve it directly from the templates/ folder.
    return app.send_static_file('index.html')


# ── Prediction form endpoint (POST) ──────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    home_team = request.form.get('home_team', '').strip()
    away_team = request.form.get('away_team', '').strip()
    neutral   = request.form.get('neutral', 'true').lower() == 'true'

    if not home_team or not away_team:
        return jsonify({'error': 'Please select both teams.'}), 400
    if home_team == away_team:
        return jsonify({'error': 'Home and away teams must be different.'}), 400

    try:
        return jsonify(predict_match(home_team, away_team, neutral=neutral))
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


# ── JSON API endpoints ────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def api_predict():
    """JSON API endpoint — useful for programmatic calls."""
    data      = request.get_json(force=True)
    home_team = data.get('home_team', '').strip()
    away_team = data.get('away_team', '').strip()
    neutral   = data.get('neutral', True)

    if not home_team or not away_team:
        return jsonify({'error': 'home_team and away_team are required'}), 400
    if home_team == away_team:
        return jsonify({'error': 'Teams must be different'}), 400

    try:
        return jsonify(predict_match(home_team, away_team, neutral=bool(neutral)))
    except ValueError as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/teams', methods=['GET'])
def api_teams():
    """Returns the full list of teams the model knows about."""
    return jsonify({'teams': ALL_TEAMS})


# ── Group stage simulation endpoint ──────────────────────────────────────────
@app.route('/api/simulate/groups', methods=['GET'])
def api_simulate_groups():
    """
    Simulates all 72 group-stage matches and returns standings + results.
    Uses predict_match (score-based outcomes).
    """
    import itertools

    groups = {
        "A": ["South Africa",           "Mexico",      "South Korea",  "Czech Republic"],
        "B": ["Bosnia and Herzegovina", "Switzerland", "Qatar",        "Canada"],
        "C": ["Brazil",                 "Morocco",     "Haiti",        "Scotland"],
        "D": ["United States",          "Paraguay",    "Turkey",       "Australia"],
        "E": ["Germany",                "Ecuador",     "Ivory Coast",  "Curacao"],
        "F": ["Netherlands",            "Sweden",      "Japan",        "Tunisia"],
        "G": ["Belgium",                "Egypt",       "Iran",         "New Zealand"],
        "H": ["Spain",                  "Uruguay",     "Saudi Arabia", "Cape Verde"],
        "I": ["France",                 "Senegal",     "Iraq",         "Norway"],
        "J": ["Argentina",              "Algeria",     "Austria",      "Jordan"],
        "K": ["Portugal",               "DR Congo",    "Colombia",     "Uzbekistan"],
        "L": ["England",                "Croatia",     "Ghana",        "Panama"],
    }

    def build_standings(group_results):
        table = {}
        for match in group_results:
            h_goals, a_goals = map(int, match['pred_score'].split(' - '))
            for team, gf, ga in [(match['home'], h_goals, a_goals), (match['away'], a_goals, h_goals)]:
                if team not in table:
                    table[team] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}
                t = table[team]
                t['P'] += 1; t['GF'] += gf; t['GA'] += ga; t['GD'] += gf - ga
                if gf > ga:    t['W'] += 1; t['Pts'] += 3
                elif gf == ga: t['D'] += 1; t['Pts'] += 1
                else:          t['L'] += 1
        return sorted(table.items(), key=lambda x: (-x[1]['Pts'], -x[1]['GD'], -x[1]['GF']))

    output = {}
    for group, teams in groups.items():
        fixtures = list(itertools.combinations(teams, 2))
        results  = []
        for h, a in fixtures:
            try:
                pred = predict_match(h, a, neutral=True)
                results.append({
                    'home': h, 'away': a,
                    'pred_score':    pred['pred_score'],
                    'home_win_prob': pred['home_win_prob'],
                    'draw_prob':     pred['draw_prob'],
                    'away_win_prob': pred['away_win_prob'],
                })
            except ValueError:
                pass
        standings = build_standings(results)
        output[group] = {
            'fixtures':  results,
            'standings': [{'team': k, **v} for k, v in standings],
        }

    return jsonify(output)


# ── Full tournament simulation endpoint ───────────────────────────────────────
@app.route('/api/simulate/tournament', methods=['GET'])
def api_simulate_tournament():
    """
    Runs the full tournament simulation (group → knockout → final).
    Returns winners at each round and the predicted champion.
    Winner determination matches notebook: score-first, probability tiebreaker.
    """
    import itertools

    groups = {
        "A": ["South Africa",           "Mexico",      "South Korea",  "Czech Republic"],
        "B": ["Bosnia and Herzegovina", "Switzerland", "Qatar",        "Canada"],
        "C": ["Brazil",                 "Morocco",     "Haiti",        "Scotland"],
        "D": ["United States",          "Paraguay",    "Turkey",       "Australia"],
        "E": ["Germany",                "Ecuador",     "Ivory Coast",  "Curacao"],
        "F": ["Netherlands",            "Sweden",      "Japan",        "Tunisia"],
        "G": ["Belgium",                "Egypt",       "Iran",         "New Zealand"],
        "H": ["Spain",                  "Uruguay",     "Saudi Arabia", "Cape Verde"],
        "I": ["France",                 "Senegal",     "Iraq",         "Norway"],
        "J": ["Argentina",              "Algeria",     "Austria",      "Jordan"],
        "K": ["Portugal",               "DR Congo",    "Colombia",     "Uzbekistan"],
        "L": ["England",                "Croatia",     "Ghana",        "Panama"],
    }

    def build_standings(group_results):
        table = {}
        for match in group_results:
            h_goals, a_goals = map(int, match['pred_score'].split(' - '))
            for team, gf, ga in [(match['home'], h_goals, a_goals), (match['away'], a_goals, h_goals)]:
                if team not in table:
                    table[team] = {'P': 0, 'W': 0, 'D': 0, 'L': 0, 'GF': 0, 'GA': 0, 'GD': 0, 'Pts': 0}
                t = table[team]
                t['P'] += 1; t['GF'] += gf; t['GA'] += ga; t['GD'] += gf - ga
                if gf > ga:    t['W'] += 1; t['Pts'] += 3
                elif gf == ga: t['D'] += 1; t['Pts'] += 1
                else:          t['L'] += 1
        return sorted(table.items(), key=lambda x: (-x[1]['Pts'], -x[1]['GD'], -x[1]['GF']))

    def simulate_round(bracket):
        results, winners, losers = [], [], []
        for m in bracket:
            try:
                r = simulate_knockout_match(m['home'], m['away'])
                results.append({
                    'home': m['home'], 'away': m['away'],
                    'pred_score':    r['pred_score'],
                    'home_win_prob': r['home_win_prob'],
                    'draw_prob':     r['draw_prob'],
                    'away_win_prob': r['away_win_prob'],
                    'winner':        r['winner'],
                })
                winners.append(r['winner'])
                losers.append(r['loser'])
            except ValueError:
                pass
        return results, winners, losers

    # Group stage
    all_standings = {}
    for g, teams in groups.items():
        res = []
        for h, a in itertools.combinations(teams, 2):
            try:
                pred = predict_match(h, a, neutral=True)
                res.append({'home': h, 'away': a, 'pred_score': pred['pred_score']})
            except ValueError:
                pass
        all_standings[g] = build_standings(res)

    group_winners = {g: s[0][0] for g, s in all_standings.items()}
    group_runners = {g: s[1][0] for g, s in all_standings.items()}

    # Best 8 third-place teams
    thirds        = [(g, s[2][0], s[2][1]) for g, s in all_standings.items() if len(s) >= 3]
    thirds_sorted = sorted(thirds, key=lambda x: (-x[2]['Pts'], -x[2]['GD'], -x[2]['GF']))
    best_8_third  = [t[1] for t in thirds_sorted[:8]]

    W, R, T = group_winners, group_runners, best_8_third

    r32_bracket = [
        {'home': W['A'], 'away': R['B']}, {'home': W['C'], 'away': R['D']},
        {'home': W['E'], 'away': R['F']}, {'home': W['G'], 'away': R['H']},
        {'home': W['I'], 'away': R['J']}, {'home': W['K'], 'away': R['L']},
        {'home': W['B'], 'away': R['A']}, {'home': W['D'], 'away': R['C']},
        {'home': W['F'], 'away': R['E']}, {'home': W['H'], 'away': R['G']},
        {'home': W['J'], 'away': R['I']}, {'home': W['L'], 'away': R['K']},
        {'home': T[0],   'away': T[7]},   {'home': T[1],   'away': T[6]},
        {'home': T[2],   'away': T[5]},   {'home': T[3],   'away': T[4]},
    ]

    r32_res, r32_w, _   = simulate_round(r32_bracket)
    r16_bracket = [{'home': r32_w[i], 'away': r32_w[i + 1]} for i in range(0, 16, 2)]
    r16_res, r16_w, _   = simulate_round(r16_bracket)
    qf_bracket  = [{'home': r16_w[i], 'away': r16_w[i + 1]} for i in range(0, 8, 2)]
    qf_res, qf_w, qf_l  = simulate_round(qf_bracket)
    sf_bracket  = [{'home': qf_w[i],  'away': qf_w[i + 1]}  for i in range(0, 4, 2)]
    sf_res, sf_w, sf_l  = simulate_round(sf_bracket)

    _, third_w, _ = simulate_round([{'home': sf_l[0], 'away': sf_l[1]}])
    _, final_w, _ = simulate_round([{'home': sf_w[0], 'away': sf_w[1]}])

    champion  = final_w[0]
    runner_up = sf_w[1] if champion == sf_w[0] else sf_w[0]
    third     = third_w[0]

    return jsonify({
        'champion':  champion,
        'runner_up': runner_up,
        'third':     third,
        'rounds': {
            'round_of_32':    r32_res,
            'round_of_16':    r16_res,
            'quarter_finals': qf_res,
            'semi_finals':    sf_res,
        },
    })


if __name__ == '__main__':
    app.run(debug=True)
