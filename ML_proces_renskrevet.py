# -*- coding: utf-8 -*-
"""Renskrevet bilagsforløb: Random Forest -> Isolation Forest.
Kør hele filen i en ny Spyder-konsol. Blokke køres derefter i rækkefølge.
Originaldata bevares. Eksplorativ genberegning, ikke en urørt sluttest.
K-means og det senere 40-feature-spor er ikke del af dette bilag.
"""
# %% 0. Pakker og indstillinger
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MAPPE = Path(__file__).resolve().parent
DATAFIL = MAPPE / 'Dataset_Machine_tool_wear.csv'
VIS_GRAFER = False  # Sæt True for også at åbne graferne i Spyder.
FIGURMAPPE = MAPPE / 'figurer'
TABELMAPPE = MAPPE / 'tabeller'
FIGURMAPPE.mkdir(exist_ok=True)
TABELMAPPE.mkdir(exist_ok=True)
FEATURES = [f'{akse}_{stat}' for akse in ['Ax2', 'Ay2', 'Az2']
            for stat in ['RMS', 'SK', 'MEAN', 'MAX', 'KURT']]
plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 300})
figurregister = []

def gem(fig, navn, beskrivelse):
    fig.tight_layout()
    fig.savefig(FIGURMAPPE / (navn + '.png'), bbox_inches='tight')
    fig.savefig(FIGURMAPPE / (navn + '.svg'), bbox_inches='tight')
    figurregister.append({'figur': navn, 'beskrivelse': beskrivelse})
    if VIS_GRAFER:
        plt.show()
    plt.close(fig)

def tabel(data, navn):
    data.to_csv(TABELMAPPE / (navn + '.csv'), index=False,
                sep=';', decimal=',', encoding='utf-8-sig')

def maal(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    return {'MAE_um': mean_absolute_error(y, pred),
            'RMSE_um': np.sqrt(mean_squared_error(y, pred)),
            'R2': r2_score(y, pred), 'Bias_um': np.mean(pred-y),
            'Pearson_r': np.corrcoef(y, pred)[0, 1]
            if np.ptp(y) > 0 and np.ptp(pred) > 0 else np.nan}

def iqr_graenser(s):
    q1, q3 = s.quantile([.25, .75])
    return q1-1.5*(q3-q1), q3+1.5*(q3-q1)

# %% 1. Indlæsning og datakontrol (beskrivende, hele datasættet)
df = pd.read_csv(DATAFIL)
if not {'tool', 'Vb', *FEATURES}.issubset(df.columns):
    raise ValueError('Der mangler krævede kolonner i datafilen.')
if not np.isfinite(df[['tool', 'Vb'] + FEATURES].to_numpy(dtype=float)).all():
    raise ValueError('Ugyldige værdier i modelinput, tool eller Vb. Undersøg dem først.')
# Bevar CSV-rækkefølgen; sorter aldrig efter Vb.
obs = df.groupby('tool', sort=False).cumcount()+1
tools = sorted(df.tool.unique())
missing = df.isna().sum()
tabel(missing.rename_axis('feature').reset_index(name='mangler'), '01_mangler')
kontrol = {'observationer': len(df), 'kolonner': len(df.columns),
           'manglende_vaerdier': int(missing.sum()),
           'berorte_raekker': int(df.isna().any(axis=1).sum()),
           'dubletter': int(df.duplicated().sum()),
           'uendelige': int(np.isinf(df.select_dtypes('number')).sum().sum())}
print('DATAKONTROL:', kontrol)
berort = df[['CV3.X_SK', 'CV3.X_KURT']].isna().any(axis=1)
antal = berort.groupby(df.tool).sum()
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(antal.index.astype(str), antal.values, color='steelblue')
ax.set(xlabel='Værktøj', ylabel='Antal berørte observationer', title='Manglende CV3.X-features')
gem(fig, '01_mangler', 'Observationer med mindst én manglende CV3.X-feature; rækker tælles kun én gang.')

# %% 2. Outliers – markering, ingen automatisk sletning
lo, hi = iqr_graenser(df.Vb)
vb_out = (df.Vb < lo) | (df.Vb > hi)
print('Vb-outliers:', int(vb_out.sum()), 'Øvre grænse:', hi)
tabel(pd.DataFrame({'tool': df.tool, 'observation': obs, 'Vb': df.Vb,
                    'IQR_markering': vb_out}), '02_vb_outliers')
fig, axes = plt.subplots(4, 4, figsize=(14, 12))
for ax, tool in zip(axes.flat, tools):
    m = df.tool == tool
    ax.plot(obs[m], df.loc[m, 'Vb'], lw=1.5)
    ax.scatter(obs[m & vb_out], df.loc[m & vb_out, 'Vb'], color='crimson', s=10)
    ax.axhline(hi, color='grey', ls='--', lw=1)
    ax.set(title=f'Værktøj {tool}', xlabel='Observation', ylabel='Vb (µm)')
for ax in list(axes.flat)[len(tools):]:
    ax.set_visible(False)
gem(fig, '02_vb_forloeb_outliers', 'Rød: IQR-markering. Stiplet: fælles øvre grænse. Observationer er ikke tid.')
fig, axes = plt.subplots(3, 3, figsize=(13, 9))
for row, tool in enumerate([14, 15, 16]):
    m = df.tool == tool
    for col, feature in enumerate(['Ay2_MEAN', 'Ay2_RMS', 'Ay2_MAX']):
        ax = axes[row, col]
        low, high = iqr_graenser(df[feature])
        flag = m & ((df[feature] < low) | (df[feature] > high))
        ax.plot(obs[m], df.loc[m, feature], lw=1)
        ax.scatter(obs[flag], df.loc[flag, feature], s=8, color='darkorange')
        ax.set(title=f'Værktøj {tool}: {feature}', xlabel='Observation', ylabel='Datasættets featureværdi')
gem(fig, '03_ay2_kontrol', 'Orange: fælles IQR-markering. Enhed beholdes uspecificeret; signalændring er ikke dokumenteret målefejl.')

# %% 3. Dataforståelse
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
counts = df.tool.value_counts().sort_index()
axes[0].bar(counts.index.astype(str), counts.values, color='steelblue')
axes[0].set(title='Observationer pr. værktøj', xlabel='Værktøj', ylabel='Antal')
axes[1].hist(df.Vb, bins=30, color='steelblue', edgecolor='white')
axes[1].set(title='Fordeling af flankeslid', xlabel='Vb (µm)', ylabel='Antal')
gem(fig, '04_datafordeling', 'Ujævn repræsentation af værktøjer og slidniveauer.')
corr = df.drop(columns=['tool', 'Vb']).corrwith(df.Vb)
sensor = [c for c in corr.index if c.startswith(('Ax', 'Ay', 'Az', 'Iu_s_', 'Iv_s_', 'CV3.S_'))]
fig, axes = plt.subplots(1, 2, figsize=(14, 12))
for ax, vals, title in [(axes[0], corr.loc[corr.abs().nlargest(15).index], '15 stærkeste korrelationer'),
                         (axes[1], corr.loc[sensor], 'Vibration og spindelstrøm')]:
    vals.sort_values().plot.barh(ax=ax, color='steelblue')
    ax.axvline(0, color='grey', lw=1)
    ax.set(title=title, xlabel='Pearson-korrelation med Vb', xlim=(-1, 1))
gem(fig, '05_korrelationer', 'Beskrivende korrelation på hele datasættet; ikke automatisk featurevalg.')
tabel(corr.rename_axis('feature').reset_index(name='Pearson_r'), '03_korrelationer')

# %% 4. Gruppeopdeling og Random Forest (samme parametre som originalen)
X, y = df[FEATURES], df.Vb
split = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
train_idx, test_idx = next(split.split(X, y, groups=df.tool))
assert not set(df.tool.iloc[train_idx]) & set(df.tool.iloc[test_idx])
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X.iloc[train_idx], y.iloc[train_idx])
pred = model.predict(X.iloc[test_idx])
resultater = df.iloc[test_idx][['tool', 'Vb']].copy()
resultater['observation'] = obs.iloc[test_idx]
resultater['forudsagt_Vb'] = pred
tabel(resultater, '04_rf_forudsigelser')
performance = [{'saet': 'traening', **maal(y.iloc[train_idx], model.predict(X.iloc[train_idx]))},
               {'saet': 'test', **maal(y.iloc[test_idx], pred)},
               {'saet': 'fast_reference_test', **maal(y.iloc[test_idx], np.full(len(test_idx), y.iloc[train_idx].mean()))}]
tabel(pd.DataFrame(performance), '05_rf_samlet')
pr_tool = pd.DataFrame([{'tool': t, **maal(g.Vb, g.forudsagt_Vb)} for t, g in resultater.groupby('tool')])
tabel(pr_tool, '06_rf_pr_tool')
print('RF:', pd.DataFrame(performance).round(3).to_string(index=False))
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
upper = max(resultater.Vb.max(), resultater.forudsagt_Vb.max())*1.05
for ax, (tool, g) in zip(axes, resultater.groupby('tool')):
    ax.scatter(g.Vb, g.forudsagt_Vb, s=12, alpha=.7)
    ax.plot([0, upper], [0, upper], '--', color='grey')
    ax.set(title=f'Værktøj {tool}', xlabel='Registreret Vb (µm)', ylabel='Forudsagt Vb (µm)', xlim=(0, upper), ylim=(0, upper))
gem(fig, '06_rf_scatter', 'Samme akser for alle testværktøjer; diagonal angiver perfekt overensstemmelse.')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
pr_tool.set_index('tool')[['MAE_um', 'RMSE_um']].plot.bar(ax=axes[0], rot=0)
axes[0].set(ylabel='Fejl (µm)', xlabel='Værktøj', title='Random Forest: testfejl')
axes[1].bar(pr_tool.tool.astype(str), pr_tool.R2, color='steelblue')
axes[1].axhline(0, color='black', lw=1)
axes[1].set(ylabel='R²', xlabel='Værktøj', title='Forklaringsgrad på testværktøjer')
gem(fig, '07_rf_fejl', 'Negativ R² betyder større kvadreret fejl end testgruppens eget Vb-gennemsnit.')

# %% 5. Diagnose – featuregennemsnit og værktøjsspecifik sammenhæng
shift = (X[df.tool == 15].mean()-X.iloc[train_idx].mean())/X.iloc[train_idx].std().replace(0, np.nan)
fig, ax = plt.subplots(figsize=(10, 6))
shift.sort_values().plot.barh(ax=ax, color='steelblue')
ax.axvline(-2, ls='--', color='grey'); ax.axvline(2, ls='--', color='grey')
ax.set(title='Værktøj 15 i forhold til træningsdata', xlabel='Forskel i gennemsnit / træningsstandardafvigelse')
gem(fig, '08_featureforskydning', 'Gennemsnit alene kan ikke udelukke forskelle i fordelinger eller samspil mellem features.')
relationer = pd.DataFrame({str(t): df.loc[df.tool == t, FEATURES].corrwith(df.loc[df.tool == t, 'Vb']) for t in [3, 13, 15]})
relationer['traening'] = X.iloc[train_idx].corrwith(y.iloc[train_idx])
tabel(relationer.rename_axis('feature').reset_index(), '07_korrelation_pr_tool')

# %% 6. Leave-one-tool-out – separat modelvariabel for hver fold
loto = []
for tool in tools:
    train_mask, test_mask = df.tool != tool, df.tool == tool
    fold_model = RandomForestRegressor(n_estimators=100, random_state=42)
    fold_model.fit(X[train_mask], y[train_mask])
    loto.append({'tool': tool, **maal(y[test_mask], fold_model.predict(X[test_mask]))})
loto = pd.DataFrame(loto)
tabel(loto, '08_leave_one_tool_out')
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(loto.tool.astype(str), loto.R2, color=['seagreen' if r > 0 else 'indianred' for r in loto.R2])
ax.axhline(0, color='black', lw=1)
ax.set(title='Random Forest: ét helt værktøj udeladt ad gangen', xlabel='Udeladt værktøj', ylabel='R²')
gem(fig, '09_leave_one_tool_out', 'Gruppebaseret krydsvalidering. Ikke uafhængig sluttest efter metodevalg.')
print('LOTO: positiv R²:', int((loto.R2 > 0).sum()), 'af', len(loto), 'median:', loto.R2.median())

# %% 7. Rækkefølgekontrol – Vb bruges her, men aldrig som IF-input
order = []
for tool, g in df.groupby('tool', sort=True):
    order.append({'tool': tool, 'Vb_start': g.Vb.iloc[0], 'Vb_slut': g.Vb.iloc[-1],
                  'antal_fald': int((g.Vb.diff() < 0).sum()),
                  'Spearman_obs_Vb': spearmanr(np.arange(len(g)), g.Vb).statistic})
order = pd.DataFrame(order)
tabel(order, '09_raekkefoelge')
# Samme afgrænsning som originalforløbet, eksplicit baseret på start-Vb.
iso_tools = order.loc[order.Vb_start == 0, 'tool'].tolist()

# %% 8. Isolation Forest og korrigeret alarmberegning
iso_dele, alarmoversigt = [], []
for tool in iso_tools:
    g = df.loc[df.tool == tool, ['tool', 'Vb'] + FEATURES].reset_index(drop=True).copy()
    n = len(g)
    nb = min(max(10, int(n*.20)), n-1)
    scaler = StandardScaler().fit(g[FEATURES].iloc[:nb])
    z = scaler.transform(g[FEATURES])
    iso_model = IsolationForest(n_estimators=300, contamination='auto', random_state=42)
    iso_model.fit(z[:nb])
    score = -iso_model.decision_function(z)
    _, threshold = iqr_graenser(pd.Series(score[:nb]))
    # Rullende vinduer beregnes KUN på observationer efter baseline.
    over = pd.Series(score[nb:] > threshold)
    alarm3 = over.rolling(3, min_periods=3).sum().eq(3).to_numpy()
    alarm57 = over.rolling(7, min_periods=7).sum().ge(5).to_numpy()
    g['observation'] = np.arange(1, n+1)
    g['baseline'] = g.observation <= nb
    g['score'] = score
    g['taerskel'] = threshold
    g['over_taerskel'] = np.r_[np.zeros(nb, dtype=bool), over.to_numpy()]
    g['alarm_3_i_traek'] = np.r_[np.zeros(nb, dtype=bool), alarm3]
    g['alarm_5_af_7'] = np.r_[np.zeros(nb, dtype=bool), alarm57]
    assert not g.loc[g.baseline, ['alarm_3_i_traek', 'alarm_5_af_7']].any().any()
    summary = {'tool': tool, 'baseline_antal': nb,
               'Spearman_Vb_score': spearmanr(g.Vb, score).statistic,
               'score_baseline_mean': score[:nb].mean(), 'score_slut_mean': score[-nb:].mean()}
    for rule in ['over_taerskel', 'alarm_3_i_traek', 'alarm_5_af_7']:
        active = g[g[rule]]
        summary[rule+'_foerste_obs'] = active.observation.iloc[0] if len(active) else np.nan
        summary[rule+'_foerste_Vb'] = active.Vb.iloc[0] if len(active) else np.nan
    alarmoversigt.append(summary)
    iso_dele.append(g)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(g.observation, g.Vb, color='steelblue', label='Vb fra datasættet')
    a = g.alarm_5_af_7
    axes[0].scatter(g.loc[a, 'observation'], g.loc[a, 'Vb'], s=10, color='crimson', label='Aktiv 5-af-7-alarm')
    axes[0].set(title=f'Isolation Forest – værktøj {tool}', ylabel='Vb (µm)')
    axes[0].legend(fontsize=9)
    axes[1].plot(g.observation, score, lw=1.2, label='Afvigelsesscore')
    axes[1].axhline(threshold, color='darkorange', ls='--', label='Q3 + 1,5 × IQR fra baseline')
    axes[1].scatter(g.loc[a, 'observation'], g.loc[a, 'score'], s=10, color='crimson')
    for ax in axes:
        ax.axvspan(.5, nb+.5, color='grey', alpha=.15)
    axes[1].set(xlabel='Observationsnummer (ikke driftstid)', ylabel='Afvigelsesscore')
    axes[1].legend(fontsize=9)
    gem(fig, f'10_IF_vaerktoej_{tool}', 'Gråt: baseline. Rød: bekræftet procesændring efter 5-af-7-reglen; ikke en valideret skifteanbefaling.')
iso_resultater = pd.concat(iso_dele, ignore_index=True)
alarmoversigt = pd.DataFrame(alarmoversigt)
tabel(iso_resultater, '10_isolation_observationer')
tabel(alarmoversigt, '11_isolation_oversigt')

# %% 9. Sammenligning af de to dokumenterede alarmregler
fig, axes = plt.subplots(5, 2, figsize=(13, 12))
for ax, (tool, g) in zip(axes.flat, iso_resultater.groupby('tool')):
    for level, rule, color in [(1, 'alarm_3_i_traek', 'steelblue'), (2, 'alarm_5_af_7', 'crimson')]:
        ax.scatter(g.loc[g[rule], 'observation'], np.full(g[rule].sum(), level), s=8, color=color)
    ax.axvspan(.5, g.loc[g.baseline, 'observation'].max()+.5, color='grey', alpha=.15)
    ax.set(title=f'Værktøj {tool}', yticks=[1, 2], yticklabels=['3 i træk', '5 af 7'],
           ylim=(.5, 2.5), xlim=(0, len(g)+1), xlabel='Observation')
gem(fig, '11_alarmregler', 'Samme score og IQR-grænse; to dokumenterede regler. Vinduer starter efter baseline.')

# %% 10. Sporbarhed og figurregister
metadata = {'data_sha256': hashlib.sha256(DATAFIL.read_bytes()).hexdigest(),
            'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'sklearn': sklearn.__version__,
            'matplotlib': matplotlib.__version__, 'features': FEATURES,
            'test_tools': sorted(df.tool.iloc[test_idx].unique().tolist()),
            'datakontrol': kontrol, 'rf_parametre': model.get_params(),
            'IF_parametre': iso_model.get_params()}
(MAPPE/'koerselsoplysninger.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
tabel(pd.DataFrame(figurregister), '12_figurregister')
print('Færdig. Figurer:', len(figurregister), 'Gemte resultater i:', MAPPE)
