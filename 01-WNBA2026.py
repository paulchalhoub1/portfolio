import streamlit as st
from nba_api.stats.endpoints import teamgamelogs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from sklearn.metrics import precision_score, recall_score
from nba_api.stats.endpoints import leaguedashteamstats

logs = teamgamelogs.TeamGameLogs(season_nullable='2026', league_id_nullable='10').get_data_frames()[0].dropna()
st.write(logs.columns)

logs['HOME_OR_AWAY'] = np.where(
    logs['MATCHUP'].str.contains(' vs. '), 1,
    np.where(logs['MATCHUP'].str.contains(' @ '), 0, np.nan)
)
logs['MATCHUP'] = logs['MATCHUP'].apply(lambda x: x.split(' ')[2])

logs['W/L'] = np.where(logs['WL'] == 'W', 1, 0)
logs = logs.drop(columns=['SEASON_YEAR', 'TEAM_NAME', 'WL', 'GP_RANK', 'W_RANK', 'L_RANK', 'W_PCT_RANK', 'MIN_RANK', 'FGM_RANK',
       'FGA_RANK', 'FG_PCT_RANK', 'FG3M_RANK', 'FG3A_RANK', 'FG3_PCT_RANK',
       'FTM_RANK', 'FTA_RANK', 'FT_PCT_RANK', 'OREB_RANK', 'DREB_RANK',
       'REB_RANK', 'AST_RANK', 'TOV_RANK', 'STL_RANK', 'BLK_RANK', 'BLKA_RANK',
       'PF_RANK', 'PFD_RANK', 'PTS_RANK', 'PLUS_MINUS_RANK', 'AVAILABLE_FLAG'])
logs = logs.sort_values(['TEAM_ID', 'GAME_DATE'])

rolling_cols = ['FGM', 'FGA', 'FG_PCT', 'FG3M',
       'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'REB', 'AST',
       'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS', 'PLUS_MINUS']

for col in rolling_cols:
    logs[f'{col}_PRE'] = (logs.groupby('TEAM_ID')[col]
                             .transform(lambda x: x.shift(1).expanding().mean()))

logs = logs.dropna(subset=[f'{col}_PRE' for col in rolling_cols])
st.write(logs)

matchups_logs = logs.merge(
    logs,
    left_on=['GAME_ID', 'TEAM_ABBREVIATION', 'MATCHUP'],
    right_on=['GAME_ID', 'MATCHUP', 'TEAM_ABBREVIATION'],
    suffixes=('', '_OPP')
)
matchups_logs = matchups_logs.drop(columns=['GAME_DATE', 'TEAM_ABBREVIATION_OPP', 'MATCHUP_OPP', 'GAME_DATE_OPP', 'GAME_ID', 'GAME_ID', 'W/L_OPP'])

all_teams = sorted(set(logs['TEAM_ABBREVIATION'].unique()) | set(logs['MATCHUP'].unique()))
team_to_id = {team: idx for idx, team in enumerate(all_teams)}
matchups_logs['TEAM_ID'] = matchups_logs['TEAM_ABBREVIATION'].map(team_to_id)
matchups_logs['MATCHUP_ID'] = matchups_logs['MATCHUP'].map(team_to_id)

st.write(matchups_logs)

df_group_one = matchups_logs[['TEAM_ABBREVIATION', 'MATCHUP', 'W/L']]
df_group_one = df_group_one.groupby(['TEAM_ABBREVIATION', 'MATCHUP'], as_index=False).agg({'W/L': 'mean'})
grouped_pivot = df_group_one.pivot(index='TEAM_ABBREVIATION', columns='MATCHUP')

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

row_labels = grouped_pivot.columns.levels[1]
col_labels = grouped_pivot.index

ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)

ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

plt.xticks(rotation=90)

fig.colorbar(im)
st.pyplot(fig)

df_group_one = matchups_logs[['FG_PCT_PRE', 'FG_PCT_PRE_OPP', 'W/L']]
df_group_one = df_group_one.groupby(['FG_PCT_PRE', 'FG_PCT_PRE_OPP'], as_index=False).agg({'W/L': 'mean'})
grouped_pivot = df_group_one.pivot(index='FG_PCT_PRE', columns='FG_PCT_PRE_OPP')

fig, ax = plt.subplots()
im = ax.pcolor(grouped_pivot, cmap='RdBu')

fig.colorbar(im)
st.pyplot(fig)

matchups_logs = matchups_logs.drop(columns=['TEAM_ABBREVIATION', 'MATCHUP'])
st.write(matchups_logs.corr())
st.write(matchups_logs.columns)

final_features = []

for col in matchups_logs.columns:
  pearson_coef, p_value = stats.pearsonr(matchups_logs[col], matchups_logs['W/L'])
  st.write(f"The Pearson Correlation Coefficient of {col} is {pearson_coef} with a P-value of P = {p_value}")
  if p_value < 0.001:
    st.write("This is a strong correlation")
    final_features.append(col)
  elif p_value < 0.05:
    st.write("This is a moderate correlation")
    final_features.append(col)
  else:
    st.write("This is a weak correlation")
  st.write()

st.write(final_features)

X = matchups_logs[['FGM_PRE', 'FG_PCT_PRE', 'FG3M_PRE', 'FG3_PCT_PRE', 'DREB_PRE', 'REB_PRE', 'AST_PRE', 'PTS_PRE', 'PLUS_MINUS_PRE', 'HOME_OR_AWAY', 'FGM_PRE_OPP', 'FG_PCT_PRE_OPP', 'FG3M_PRE_OPP', 'FG3_PCT_PRE_OPP', 'DREB_PRE_OPP', 'REB_PRE_OPP', 'AST_PRE_OPP', 'PTS_PRE_OPP', 'PLUS_MINUS_PRE_OPP', 'HOME_OR_AWAY_OPP']]
y_wl = matchups_logs['W/L']

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

x_train, x_test, y_train, y_test = train_test_split(X, y_wl, test_size=0.3, random_state=42)

models = [LogisticRegression(max_iter=1000, random_state=42), SVC(
  kernel='linear', random_state=42, probability=True), XGBClassifier(booster='gblinear', random_state=42), RandomForestClassifier(random_state=42)]

for model in models:
  model.fit(x_train, y_train)
  accuracy = model.score(x_test, y_test)
  y_pred = model.predict(x_test)
  cv_accuracy  = cross_val_score(model, X, y_wl, cv=5, scoring='accuracy')
  cv_precision = cross_val_score(model, X, y_wl, cv=5, scoring='precision')
  cv_recall    = cross_val_score(model, X, y_wl, cv=5, scoring='recall')

  st.write(f'{model} : ')
  st.write("Accuracy: ", accuracy)
  st.write("Precision: ", precision_score(y_test, y_pred))
  st.write("Recall: ", recall_score(y_test, y_pred))
  st.write(f"CV Accuracy:  {cv_accuracy.mean():.3f} (+/- {cv_accuracy.std():.3f})")
  st.write(f"CV Precision: {cv_precision.mean():.3f} (+/- {cv_precision.std():.3f})")
  st.write(f"CV Recall:    {cv_recall.mean():.3f} (+/- {cv_recall.std():.3f})")
  st.write('------------------------------------------')

  ConfusionMatrixDisplay.from_estimator(model, x_test, y_test)
  st.pyplot()
  RocCurveDisplay.from_estimator(model, x_test, y_test)
  st.pyplot()
  PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
  st.pyplot()

team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2026', league_id_nullable='10', per_mode_detailed='PerGame').get_data_frames()[0]
st.write(team_stats)

cols_to_scale = ['FGM', 'FG3M', 'DREB', 'REB', 'AST', 'PTS']
for col in team_stats.columns:
  team_stats[f'{col}_PRE'] = team_stats[col]

matchups = [[11, 9], [2, 4]]

clf = XGBClassifier(booster='gblinear', random_state=42)
clf.fit(X, y_wl)

log = LogisticRegression(max_iter=1000, random_state=42)
log.fit(X, y_wl)

svc = SVC(kernel='linear', probability=True, random_state=42)
svc.fit(X, y_wl)

rf = RandomForestClassifier(random_state=42)
rf.fit(X, y_wl)

for i, matchup in enumerate(matchups):

  away = team_stats.iloc[[matchup[0]]].copy()
  home = team_stats.iloc[[matchup[1]]].copy()

  away['HOME_OR_AWAY'] = 0
  home['HOME_OR_AWAY'] = 1

  away = away.reset_index(drop=True)
  home = home.reset_index(drop=True)


  X_new1 = pd.concat([away, home.add_suffix('_OPP')], axis=1)
  X_new1 = X_new1.drop(columns=[c for c in X_new1.columns if c not in X.columns])

  X_new2 = pd.concat([home, away.add_suffix('_OPP')], axis=1)
  X_new2 = X_new2.drop(columns=[c for c in X_new2.columns if c not in X.columns])

  X_new = pd.concat([X_new1, X_new2]).reset_index(drop=True)
  X_new = pd.DataFrame(scaler.fit_transform(X_new), columns=X_new.columns)

  away_name = away['TEAM_NAME'].values[0]
  home_name = home['TEAM_NAME'].values[0]

  models = {'XGBoost': clf, 'Logistic Regression': log, 'SVC': svc, 'Random Forest': rf}

  st.write(f"=== {away_name} (away) vs {home_name} (home) ===")
  for name, model in models.items():
      probs = model.predict_proba(X_new)[:, 1]
      away_win_prob = (probs[0] + (1 - probs[1])) / 2
      home_win_prob = (probs[1] + (1 - probs[0])) / 2
      st.write(f"  {name}: {away_name} {away_win_prob*100:.1f}% | {home_name} {home_win_prob*100:.1f}%")
  st.write()
