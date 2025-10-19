import os, glob, time, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

# =========================
# Configuración
# =========================
BASE_PATH = os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', 'fraud_stream_parquet')
OUT_DIR   = os.path.join(os.environ.get('USERPROFILE', ''), 'Downloads', 'baseline1_results')
os.makedirs(OUT_DIR, exist_ok=True)

FEATURES = [
    'TX_AMOUNT', 'TX_TIME_DAYS', 'TX_TIME_SECONDS',
    'x_customer_id','y_customer_id','mean_amount','std_amount','mean_nb_tx_per_day',
    'x_terminal_id','y_terminal_id'
]
TARGET = 'TX_FRAUD'

PRE_MONTHS = 4                 # meses para pretraining
VAL_DAYS_LAST_MONTH = 14       # días del mes 4 para calibrar umbral F1
GRANULARITY = 'month'          # 'month' (recomendado). Si quieres diario: 'day'.
TIMELINE_FILE = os.path.join(BASE_PATH, 'timeline.parquet')  # opcional; si no existe, TTA90 usa un solo régimen

# TTA90: ventanas para "plateau" por régimen (si GRANULARITY='month', usa 2; si 'day', usa 10)
PLATEAU_WINDOW = 2 if GRANULARITY == 'month' else 10
TTA_TARGET_FRAC = 0.90  # 90%

# =========================
# Utilidades de carga/orden
# =========================
def load_all_parquet_by_year(base_path: str) -> pd.DataFrame:
    year_dirs = sorted(glob.glob(os.path.join(base_path, "TX_YEAR=*")))
    dfs = []
    for ydir in year_dirs:
        files = sorted(glob.glob(os.path.join(ydir, "*.parquet")))
        if not files:
            continue
        dfs.append(pd.concat([pd.read_parquet(f) for f in files], ignore_index=True))
    if not dfs:
        raise RuntimeError(f"No se encontraron Parquet en {base_path}")
    df = pd.concat(dfs, ignore_index=True)
    # Orden temporal sin TX_DATETIME
    df = df.sort_values(['TX_YEAR','TX_MONTH','TX_DAY','TX_TIME_SECONDS'], kind='mergesort').reset_index(drop=True)
    return df

def add_time_indexes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    first_year = int(df['TX_YEAR'].min())
    df['_month_idx'] = (df['TX_YEAR'] - first_year) * 12 + df['TX_MONTH']  # 1..24
    # índice de "día absoluto" desde el inicio (por si usas diario)
    # aproximación: mapeo (year,month,day) a orden relativo creciente
    df['_day_abs'] = (df['_month_idx'] - 1) * 31 + df['TX_DAY']  # suficiente para ordenar de forma estable
    return df

def group_chunks(df: pd.DataFrame, granularity='month'):
    if granularity == 'month':
        # devuelve (clave, df_mes, etiqueta_eje)
        for (y, m), g in df.groupby(['TX_YEAR','TX_MONTH'], sort=True):
            yield (int(y), int(m)), g, f"{int(y)}-{int(m):02d}"
    elif granularity == 'day':
        # etiqueta densa por día (dentro de cada mes)
        for (y, m, d), g in df.groupby(['TX_YEAR','TX_MONTH','TX_DAY'], sort=True):
            yield (int(y), int(m), int(d)), g, f"{int(y)}-{int(m):02d}-{int(d):02d}"
    else:
        raise ValueError("granularity debe ser 'month' o 'day'.")

# =========================
# Split pretraining/val
# =========================
def split_pretrain_val(df: pd.DataFrame, pre_months=4, val_days_last_month=14):
    df = df.copy()
    m = pre_months
    pre_mask = (df['_month_idx'] < m)              # meses 1..3
    last_month_mask = (df['_month_idx'] == m)      # mes 4
    # últimas 2 semanas del mes 4 como val
    dmax = int(df.loc[last_month_mask, 'TX_DAY'].max())
    cutoff = max(1, dmax - val_days_last_month + 1)
    train_last = last_month_mask & (df['TX_DAY'] < cutoff)
    val_last   = last_month_mask & (df['TX_DAY'] >= cutoff)
    train_mask = pre_mask | train_last
    val_mask   = val_last
    return train_mask, val_mask

# =========================
# Modelo base
# =========================
def fit_baseline(train_df, val_df, features, target='TX_FRAUD'):
    # estandarización + LR balanceada
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train_df[features].astype('float32').values)
    ytr = train_df[target].astype('int32').values

    clf = LogisticRegression(max_iter=1000, class_weight='balanced', n_jobs=-1, random_state=42)
    t0 = time.perf_counter()
    clf.fit(Xtr, ytr)
    train_time_s = time.perf_counter() - t0

    # calibración de umbral F1 en el val del mes 4
    Xv = scaler.transform(val_df[features].astype('float32').values)
    yv = val_df[target].astype('int32').values

    t0 = time.perf_counter()
    pv = clf.predict_proba(Xv)[:,1]
    inf_time_val_s = time.perf_counter() - t0
    ap_ref = float(average_precision_score(yv, pv))

    prec, rec, thr = precision_recall_curve(yv, pv)
    f1s = (2 * prec * rec) / (prec + rec + 1e-12)
    best_idx = int(np.nanargmax(f1s[:-1])) if len(thr) > 0 else 0
    best_thr = float(thr[best_idx]) if len(thr) > 0 else 0.5
    f1_ref = float(f1_score(yv, (pv >= best_thr).astype(int)))

    return scaler, clf, best_thr, ap_ref, f1_ref, train_time_s, inf_time_val_s / max(1, len(yv))

# =========================
# Evaluación serie temporal
# =========================
def evaluate_stream(df, features, scaler, clf, thr, granularity='month'):
    rows = []
    for key, chunk, label in group_chunks(df, granularity=granularity):
        X = scaler.transform(chunk[features].astype('float32').values)
        y = chunk[TARGET].astype('int32').values

        t0 = time.perf_counter()
        p = clf.predict_proba(X)[:,1]
        dt = time.perf_counter() - t0
        ap  = float(average_precision_score(y, p))
        f1  = float(f1_score(y, (p >= thr).astype(int)))
        ms_per_tx = (dt / max(1, len(y))) * 1e3

        rec = {
            "key": key, "label": label,
            "n": int(len(y)),
            "AUPRC": ap, "F1": f1,
            "infer_ms_per_tx": ms_per_tx
        }
        rows.append(rec)

    metrics = pd.DataFrame(rows).sort_values('label').reset_index(drop=True)
    return metrics

# =========================
# Adaptación (TTA90) y Olvido (MaxDrawdown%)
# =========================
def load_timeline_month_states(timeline_path, df_all, granularity='month'):
    """
    Devuelve lista de tuplas (start_idx, end_idx, state_str) por régimen.
    Si no hay timeline, asume un único régimen (todo el periodo).
    """
    # Mapeo etiqueta->índice en la serie de métricas
    if granularity != 'month':
        # Para granularidad diaria, necesitarías timeline diario (S1,S2,S3 por día)
        # Este script centra TTA90 mensual porque tus escenarios están definidos en meses/bimestres.
        return [(0, None, "ALL")]  # todo el periodo

    # Si no existe timeline -> todo el periodo
    if not os.path.exists(timeline_path):
        return [(0, None, "ALL")]

    tl = pd.read_parquet(timeline_path)
    # timeline esperado con columnas: TX_TIME_DAYS, S1,S2,S3 (0/1), state (string)
    # agregamos por mes (modo más frecuente del estado en el mes)
    # necesitamos mapear mes: usaremos df_all para (year, month) por day_idx
    aux = df_all[['TX_YEAR','TX_MONTH','TX_DAY']].drop_duplicates().copy()
    # aproximación: convertir (year,month,day) a day_idx relativo al start_date original si lo tuviéramos;
    # como no lo tenemos aquí, asumimos que timeline.TX_TIME_DAYS arranca en 0 y que df_all está ordenado.
    # Para emparejar, sacamos el mes desde TX_TIME_DAYS basándonos en el primer año/mes/día del df.
    # Mas simple: construimos un DataFrame de meses con rango de TX_TIME_DAYS (si no es trivial, caemos a "ALL").
    if 'TX_TIME_DAYS' not in tl.columns:
        return [(0, None, "ALL")]

    # mes por TX_TIME_DAYS usando df_all (aprox): tomamos el valor mínimo de TX_TIME_DAYS para cada (Y,M)
    # y asignamos días dentro de ese bloque.
    # Si no hay correspondencia exacta, reducimos a "ALL".
    try:
        # Mapeo día->(Y,M)
        # Creamos un mapping rápido usando el df_all (ordenado) tomando pares (day_abs -> month label)
        df_sorted = df_all.sort_values(['TX_YEAR','TX_MONTH','TX_DAY']).copy()
        df_sorted['_day_seq'] = np.arange(len(df_sorted))
        # day_seq no es TX_TIME_DAYS, pero guarda orden; usamos cortes por mes para generar ranges
        month_groups = df_sorted.groupby(['TX_YEAR','TX_MONTH'])['_day_seq'].agg(['min','max']).reset_index()
        # timeline por TX_TIME_DAYS -> lo llevamos a day_seq asumiendo igualdad de longitud total
        max_tl = int(tl['TX_TIME_DAYS'].max())
        max_seq = int(df_sorted['_day_seq'].max())
        # Escala lineal day_idx ~ TX_TIME_DAYS (aprox)
        tl['_day_seq'] = (tl['TX_TIME_DAYS'] * (max_seq / max(1, max_tl))).round().astype(int).clip(0, max_seq)
        # asigna mes por _day_seq usando interval merge
        month_ranges = []
        for _, r in month_groups.iterrows():
            month_ranges.append( (int(r['min']), int(r['max']), int(r['TX_YEAR']), int(r['TX_MONTH'])) )
        month_ranges = sorted(month_ranges)

        def seq_to_ym(s):
            # busca en month_ranges
            # (búsqueda lineal está bien: solo 24 meses)
            for mn, mx, y, m in month_ranges:
                if mn <= s <= mx:
                    return y, m
            return None, None

        tl[['TL_YEAR','TL_MONTH']] = tl['_day_seq'].apply(lambda s: pd.Series(seq_to_ym(s)))

        tlm = tl.dropna(subset=['TL_YEAR','TL_MONTH']).copy()
        tlm['state'] = tlm.apply(lambda r: f"S1={int(r.get('S1',0))}|S2={int(r.get('S2',0))}|S3={int(r.get('S3',0))}", axis=1)

        # estado por mes = moda
        month_state = (tlm
            .groupby(['TL_YEAR','TL_MONTH'])['state']
            .agg(lambda s: s.value_counts().idxmax())
            .reset_index()
            .sort_values(['TL_YEAR','TL_MONTH'])
        )

        # convertir a segmentos contiguos por estado
        labels = [f"{int(y)}-{int(m):02d}" for y,m in zip(month_state['TL_YEAR'], month_state['TL_MONTH'])]
        states = month_state['state'].tolist()

        segments = []
        if labels:
            start = 0
            for i in range(1, len(labels)):
                if states[i] != states[i-1]:
                    segments.append( (start, i-1, states[i-1]) )
                    start = i
            segments.append( (start, len(labels)-1, states[-1]) )
        else:
            segments = [(0, None, "ALL")]
        # devolvemos índices relativos al primer mes presente en métricas
        return segments
    except Exception:
        # si algo no calza, usamos régimen único
        return [(0, None, "ALL")]

def compute_max_drawdown_percent(series_ap):
    running_max = -np.inf
    max_dd = 0.0
    for ap in series_ap:
        running_max = max(running_max, ap)
        if running_max > 0:
            dd = (ap - running_max) / running_max
            max_dd = min(max_dd, dd)
    return 100.0 * max_dd  # negativo o 0

def compute_tta90_per_regime(metrics_df, segments, window=2, target_frac=0.90):
    """
    segments: lista de (start_idx, end_idx, state_str) sobre la serie mensual de metrics_df.
    Devuelve DataFrame con TTA90 por segmento (en unidades de puntos de la serie = meses).
    """
    # Usamos la columna 'AUPRC' y asumimos metrics_df ordenado por 'label' ascendente
    ap = metrics_df['AUPRC'].values
    res = []
    # Si segments es ALL, consideramos un solo régimen: todo el rango
    if len(segments) == 1 and segments[0][2] == "ALL":
        start = 0
        end = len(ap) - 1
        plateau = ap[max(start, end - window + 1): end + 1].mean() if end >= start else np.nan
        thr = target_frac * plateau if not math.isnan(plateau) else np.nan
        tta = None
        if not math.isnan(thr):
            tta = None
            for i in range(start, end + 1):
                if ap[i] >= thr:
                    tta = i - start  # meses desde inicio de régimen
                    break
        res.append({"segment": "ALL", "start_idx": start, "end_idx": end,
                    "plateau": plateau, "thr": thr, "TTA90": tta})
        return pd.DataFrame(res)

    # Si hay segmentos mensuales (con estados)
    # NOTA: metrics_df podría empezar en mes 5; nuestros segments asumen índice 0 en el primer mes present
    # Tomamos intersección segura con el rango de metrics
    total_n = len(ap)
    for (start, end, state) in segments:
        if end is None:
            end = total_n - 1
        start = max(0, start)
        end   = min(total_n - 1, end)
        if end < start:
            continue
        plateau = ap[max(start, end - window + 1): end + 1].mean()
        thr = target_frac * plateau
        tta = None
        for i in range(start, end + 1):
            if ap[i] >= thr:
                tta = i - start
                break
        res.append({"segment": state, "start_idx": start, "end_idx": end,
                    "plateau": plateau, "thr": thr, "TTA90": tta})
    return pd.DataFrame(res)

# =========================
# Gráficas
# =========================
def plot_series(metrics_df, out_dir):
    # AUPRC en el tiempo
    plt.figure()
    plt.plot(metrics_df['label'], metrics_df['AUPRC'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Periodo'); plt.ylabel('AUPRC'); plt.title('AUPRC en el tiempo (baseline #1)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'auprc_tiempo.png')); plt.close()

    # F1 en el tiempo
    plt.figure()
    plt.plot(metrics_df['label'], metrics_df['F1'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Periodo'); plt.ylabel('F1 (umbral fijo)'); plt.title('F1 en el tiempo (baseline #1)')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'f1_tiempo.png')); plt.close()

    # AUPRC y running max (para visualizar drawdown)
    running_max = np.maximum.accumulate(metrics_df['AUPRC'].values)
    plt.figure()
    plt.plot(metrics_df['label'], metrics_df['AUPRC'], label='AUPRC')
    plt.plot(metrics_df['label'], running_max, label='Pico acumulado', linestyle='--')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Periodo'); plt.ylabel('AUPRC'); plt.title('AUPRC vs. Pico acumulado')
    plt.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'auprc_drawdown.png')); plt.close()

    # Latencia aprox por transacción
    plt.figure()
    plt.plot(metrics_df['label'], metrics_df['infer_ms_per_tx'])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Periodo'); plt.ylabel('ms por transacción (aprox)'); plt.title('Latencia de inferencia p95 aprox')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'latencia_tiempo.png')); plt.close()

def plot_tta90_bars(tta_df, out_dir):
    # Bar chart por régimen
    labels = []
    values = []
    for _, r in tta_df.iterrows():
        lab = r['segment']
        if lab == 'ALL': lab = 'Régimen único'
        labels.append(lab)
        values.append(np.nan if r['TTA90'] is None else r['TTA90'])
    plt.figure()
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels, rotation=45, ha='right')
    plt.ylabel('Meses hasta 90% del plateau'); plt.title('TTA90 por régimen')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, 'tta90_regimenes.png')); plt.close()

# =========================
# Main
# =========================
def main():
    print("Cargando datos...")
    df = load_all_parquet_by_year(BASE_PATH)
    df = add_time_indexes(df)

    # Split pretraining (meses 1..4)
    train_mask, val_mask = split_pretrain_val(df, pre_months=PRE_MONTHS, val_days_last_month=VAL_DAYS_LAST_MONTH)
    train_df = df[train_mask]
    val_df   = df[val_mask]

    # Entrenar baseline
    print("Entrenando baseline (LR) con meses 1..4...")
    features = [c for c in FEATURES if c in df.columns]
    scaler, clf, thr, ap_ref, f1_ref, train_time_s, inf_time_per_tx_val = fit_baseline(train_df, val_df, features)

    # Serie mensual desde el mes 5 (o desde donde quieras evaluar)
    print("Evaluando serie temporal...")
    eval_df = df[df['_month_idx'] >= (PRE_MONTHS + 1)]
    metrics = evaluate_stream(eval_df, features, scaler, clf, thr, granularity=GRANULARITY)

    # Guardar métricas crudas
    metrics.to_csv(os.path.join(OUT_DIR, 'monthly_metrics.csv'), index=False)

    # Olvido (MaxDrawdown%)
    mdd = compute_max_drawdown_percent(metrics['AUPRC'].values)

    # Adaptación (TTA90)
    segments = load_timeline_month_states(TIMELINE_FILE, df, granularity=GRANULARITY)
    tta_df = compute_tta90_per_regime(metrics, segments, window=PLATEAU_WINDOW, target_frac=TTA_TARGET_FRAC)
    tta_df.to_csv(os.path.join(OUT_DIR, 'tta90_por_regimen.csv'), index=False)

    # Gráficas
    plot_series(metrics, OUT_DIR)
    plot_tta90_bars(tta_df, OUT_DIR)

    # Resumen
    print("\n=== RESUMEN BASELINE #1 ===")
    print(f"AUPRC de referencia (val mes 4): {ap_ref:.4f}")
    print(f"F1 de referencia (val mes 4)   : {f1_ref:.4f} @thr={thr:.3f}")
    print(f"Tiempo de entrenamiento (s)     : {train_time_s:.2f}")
    print(f"Inferencia val (ms/tx aprox)    : {inf_time_per_tx_val*1e3:.3f}")
    print(f"MaxDrawdown% (olvido)           : {mdd:.2f}%")
    if not tta_df.empty:
        print("\nTTA90 por régimen (meses hasta 90% del plateau):")
        print(tta_df[['segment','TTA90']])

    print(f"\nArchivos guardados en: {OUT_DIR}")
    print(" - monthly_metrics.csv")
    print(" - tta90_por_regimen.csv")
    print(" - auprc_tiempo.png, f1_tiempo.png, auprc_drawdown.png, latencia_tiempo.png, tta90_regimenes.png")

if __name__ == "__main__":
    main()
