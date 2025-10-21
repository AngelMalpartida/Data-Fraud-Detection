# Adaptación completa del notebook SimulatedDataset.ipynb a un script Python

import os
import datetime
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gc
from math import ceil

# ---- Tipos/dtypes para bajar memoria ----
DTYPES = {
    "CUSTOMER_ID": "int32",
    "TERMINAL_ID": "int32",
    "TX_TIME_SECONDS": "int32",
    "TX_TIME_DAYS": "int32",
    "TX_AMOUNT": "float32",
    # Se agregarán más abajo tras el merge
}
def _ym_index(y, m):
    # índice total ordenable por (año, mes)
    return int(y) * 12 + int(m)


def month_range_mask(df, start=None, end=None):
    """
    Devuelve:
      mask_month: máscara booleana por (TX_YEAR, TX_MONTH) en [start..end] inclusive
      day_min, day_max: min y max de TX_TIME_DAYS dentro de ese rango (o None si vacío)
    start/end: dicts {"year":YYYY, "month":MM} o None.
    """
    ym = df["TX_YEAR"] * 12 + df["TX_MONTH"]
    sidx = -10**9 if start is None else _ym_index(start["year"], start["month"])
    eidx =  10**9 if end   is None else _ym_index(end["year"],   end["month"])
    mask = (ym >= sidx) & (ym <= eidx)
    if mask.any():
        dmin = int(df.loc[mask, "TX_TIME_DAYS"].min())
        dmax = int(df.loc[mask, "TX_TIME_DAYS"].max())
    else:
        dmin, dmax = None, None
    return mask, dmin, dmax

def add_time_columns(df, start_date="2025-01-01"):
    df = df.copy()
    if "TX_DATETIME" not in df:
        df["TX_DATETIME"] = pd.to_datetime(df["TX_TIME_SECONDS"], unit="s", origin=start_date)
    df["TX_DATE"]  = df["TX_DATETIME"].dt.date.astype("datetime64[ns]")
    df["TX_YEAR"]  = df["TX_DATETIME"].dt.year.astype("int16")
    df["TX_MONTH"] = df["TX_DATETIME"].dt.month.astype("int8")
    df["TX_DAY"]   = df["TX_DATETIME"].dt.day.astype("int8")
    return df

def optimize_dtypes(df):
    for c, t in DTYPES.items():
        if c in df.columns:
            df[c] = df[c].astype(t, copy=False)
    return df

def combine_profiles(transactions_df, customer_profiles_table, terminal_profiles_table, start_date="2025-01-01"):
    """
    Devuelve un DF transaccional enriquecido con ~15 numéricas:
    - TX_AMOUNT, TX_TIME_SECONDS, TX_TIME_DAYS
    - Cliente: x/y, mean_amount, std_amount, mean_nb_tx_per_day
    - Terminal: x/y
    (+ etiqueta y escenario si ya existen)
    """
    # Selecciona columnas relevantes de perfiles
    cust_cols = ["CUSTOMER_ID", "x_customer_id", "y_customer_id", 
                 "mean_amount", "std_amount", "mean_nb_tx_per_day"]
    term_cols = ["TERMINAL_ID", "x_terminal_id", "y_terminal_id"]

    df = transactions_df.merge(customer_profiles_table[cust_cols], on="CUSTOMER_ID", how="left")
    df = df.merge(terminal_profiles_table[term_cols], on="TERMINAL_ID", how="left")

    # Asegura columnas de etiqueta
    if "TX_FRAUD" not in df.columns:
        df["TX_FRAUD"] = 0
    if "TX_FRAUD_SCENARIO" not in df.columns:
        df["TX_FRAUD_SCENARIO"] = 0

    # Tiempos auxiliares (para particionar/guardar)
    df = add_time_columns(df, start_date=start_date)

    # Optimiza dtypes
    df = optimize_dtypes(df)

    return df

def add_fraud_scenario_1(df, amount_threshold=220.0,
                         start=None, end=None,   # dicts {"year":..,"month":..}
                         start_day=None, end_day=None):
    """
    Regla simple por monto. Ahora limitado por (año, mes) o por días.
    """
    df = df.copy()
    if start is not None or end is not None:
        mask_month, _, _ = month_range_mask(df, start, end)
    else:
        mask_month = pd.Series(True, index=df.index)

    # (si te pasan day-range explícito, lo intersectas)
    mask_day = pd.Series(True, index=df.index)
    if start_day is not None: mask_day &= df["TX_TIME_DAYS"] >= int(start_day)
    if end_day   is not None: mask_day &= df["TX_TIME_DAYS"] <= int(end_day)

    mask = mask_month & mask_day & (df["TX_AMOUNT"] > float(amount_threshold))
    df.loc[mask, "TX_FRAUD"] = 1
    df.loc[mask, "TX_FRAUD_SCENARIO"] = 1
    return df

def add_fraud_scenario_2(df, terminal_profiles_table,
                         n_per_day=2, window_days=28, seed=0,
                         start=None, end=None,
                         start_day=None, end_day=None):
    df = df.copy()
    if start is not None or end is not None:
        mask_month, dmin, dmax = month_range_mask(df, start, end)
        if dmin is None:
            return df
        day_list = np.sort(df.loc[mask_month, "TX_TIME_DAYS"].unique())
    else:
        dmin = 0 if start_day is None else int(start_day)
        dmax = int(df["TX_TIME_DAYS"].max()) if end_day is None else int(end_day)
        day_list = np.arange(dmin, dmax+1)

    term_ids = terminal_profiles_table["TERMINAL_ID"].values

    for day_np in day_list:
        day = int(day_np)                           # <-- clave
        rng_day = np.random.RandomState(day)        # ok con int nativo
        k = min(int(n_per_day), len(term_ids))
        if k <= 0:
            continue
        compromised = rng_day.choice(term_ids, size=k, replace=False)
        mask = (
            (df["TX_TIME_DAYS"] >= day) &
            (df["TX_TIME_DAYS"] < day + int(window_days)) &
            (df["TERMINAL_ID"].isin(compromised))
        )
        if start is not None or end is not None:
            mask &= mask_month
        df.loc[mask, "TX_FRAUD"] = 1
        df.loc[mask, "TX_FRAUD_SCENARIO"] = 2
    return df


def add_fraud_scenario_3(df, customer_profiles_table,
                         n_customers_per_day=3, window_days=14,
                         amp_factor=5.0, frac_to_flip=1/3, seed=0,
                         start=None, end=None,
                         start_day=None, end_day=None):
    df = df.copy()
    if start is not None or end is not None:
        mask_month, dmin, dmax = month_range_mask(df, start, end)
        if dmin is None:
            return df
        day_list = np.sort(df.loc[mask_month, "TX_TIME_DAYS"].unique())
    else:
        dmin = 0 if start_day is None else int(start_day)
        dmax = int(df["TX_TIME_DAYS"].max()) if end_day is None else int(end_day)
        day_list = np.arange(dmin, dmax+1)
        mask_month = pd.Series(True, index=df.index)

    cust_ids = customer_profiles_table["CUSTOMER_ID"].values
    seed = int(seed)                                # <-- por si viene no-int

    for day_np in day_list:
        day = int(day_np)                           # <-- clave
        rng_day = random.Random(day + seed)         # ahora sí: int nativo
        k = min(int(n_customers_per_day), len(cust_ids))
        if k <= 0:
            continue
        compromised = rng_day.sample(list(cust_ids), k=k)

        mask_win = (
            (df["TX_TIME_DAYS"] >= day) &
            (df["TX_TIME_DAYS"] < day + int(window_days)) &
            (df["CUSTOMER_ID"].isin(compromised)) &
            mask_month
        )
        idxs = df.index[mask_win]
        if len(idxs) == 0:
            continue

        df.loc[idxs, "TX_AMOUNT"] = (df.loc[idxs, "TX_AMOUNT"] * float(amp_factor)).astype("float32")

        nflip = max(1, int(len(idxs) * float(frac_to_flip)))
        flip_idxs = rng_day.sample(list(idxs), k=nflip)
        df.loc[flip_idxs, "TX_FRAUD"] = 1
        df.loc[flip_idxs, "TX_FRAUD_SCENARIO"] = 3
    return df


def apply_fraud_schedule(df, customer_profiles_table, terminal_profiles_table, schedule):
    """
    Aplica una lista de pasos con rango por (año, mes).
    Cada item: {"scenario": 1|2|3, "start": {"year":..,"month":..}, "end": {...}, "params": {...}}
    """
    out = df.copy()
    # Asegúrate de tener columnas de destino
    if "TX_FRAUD" not in out.columns:
        out["TX_FRAUD"] = 0
    if "TX_FRAUD_SCENARIO" not in out.columns:
        out["TX_FRAUD_SCENARIO"] = 0

    for step in schedule:
        sc = int(step["scenario"])
        start = step.get("start", None)
        end   = step.get("end", None)
        params = step.get("params", {}) or {}

        if sc == 1:
            out = add_fraud_scenario_1(out, start=start, end=end, **params)
        elif sc == 2:
            out = add_fraud_scenario_2(out, terminal_profiles_table, start=start, end=end, **params)
        elif sc == 3:
            out = add_fraud_scenario_3(out, customer_profiles_table, start=start, end=end, **params)
        else:
            print(f"[WARN] Escenario no reconocido: {sc}")
    return out


def write_parquet_by_chunks(df, base_path, freq="MS", compression="snappy"):
   
    os.makedirs(base_path, exist_ok=True)

    df = df.sort_values(
        by=["TX_YEAR","TX_MONTH","TX_DAY","TX_TIME_SECONDS"],
        kind="mergesort"
    )

    for y, g in df.groupby("TX_YEAR", sort=True):
        if g.empty:
            continue

        out_dir = os.path.join(base_path, f"TX_YEAR={int(y)}")
        os.makedirs(out_dir, exist_ok=True)

        fn = os.path.join(out_dir, f"transactions_{int(y)}.parquet")
        g.to_parquet(fn, engine="pyarrow", compression=compression, index=False)

        del g
        gc.collect()


# Generación de perfiles de clientes
def generate_customer_profiles_table(n_customers, random_state=0):
    np.random.seed(random_state)
    customer_id_properties = []
    for customer_id in range(n_customers):
        x_customer_id = np.random.uniform(0, 100)
        y_customer_id = np.random.uniform(0, 100)
        mean_amount = np.random.uniform(5, 100)
        std_amount = mean_amount / 2
        mean_nb_tx_per_day = np.random.uniform(0, 4)
        customer_id_properties.append([
            customer_id, x_customer_id, y_customer_id, mean_amount, std_amount, mean_nb_tx_per_day
        ])
    return pd.DataFrame(customer_id_properties, columns=[
        'CUSTOMER_ID', 'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day'
    ])

# Generación de perfiles de terminales
def generate_terminal_profiles_table(n_terminals, random_state=0):
    np.random.seed(random_state)
    terminal_id_properties = []
    for terminal_id in range(n_terminals):
        x_terminal_id = np.random.uniform(0, 100)
        y_terminal_id = np.random.uniform(0, 100)
        terminal_id_properties.append([terminal_id, x_terminal_id, y_terminal_id])
    return pd.DataFrame(terminal_id_properties, columns=['TERMINAL_ID', 'x_terminal_id', 'y_terminal_id'])

# Asociación de clientes a terminales
def get_list_terminals_within_radius(customer_profile, x_y_terminals, r):
    x_y_customer = customer_profile[['x_customer_id', 'y_customer_id']].values.astype(float)
    squared_diff_x_y = np.square(x_y_customer - x_y_terminals)
    dist_x_y = np.sqrt(np.sum(squared_diff_x_y, axis=1))
    return list(np.where(dist_x_y < r)[0])

# Generación de transacciones
def generate_transactions_table(customer_profile, start_date="2025-01-01", nb_days=10):
    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))
    for day in range(nb_days):
        nb_tx = np.random.poisson(customer_profile.mean_nb_tx_per_day)
        if nb_tx > 0:
            for tx in range(nb_tx):
                time_tx = int(np.random.normal(86400 / 2, 20000))
                if 0 < time_tx < 86400:
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)
                    amount = np.round(amount, decimals=2)
                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                        customer_transactions.append([
                            time_tx + day * 86400, day, customer_profile.CUSTOMER_ID, terminal_id, amount
                        ])
    customer_transactions = pd.DataFrame(customer_transactions, columns=[
        'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'
    ])
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date)
        customer_transactions = customer_transactions[['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']]
    return customer_transactions


def generate_transactions_table_seasonal(customer_profile, start_date="2025-01-01", nb_days=10,
                                         tx_mult=None, amount_mult=None, amount_noise=None):
    """
    Igual a generate_transactions_table, pero aplica factores diarios:
    - nb_tx ~ Poisson(mean_nb_tx_per_day * tx_mult[day])
    - TX_AMOUNT *= amount_mult[day] * (1 + Normal(0, amount_noise[day]))
    """
    import numpy as np, random, pandas as pd

    if tx_mult is None:       tx_mult = np.ones(nb_days, dtype="float32")
    if amount_mult is None:   amount_mult = np.ones(nb_days, dtype="float32")
    if amount_noise is None:  amount_noise = np.zeros(nb_days, dtype="float32")

    customer_transactions = []
    random.seed(int(customer_profile.CUSTOMER_ID))
    np.random.seed(int(customer_profile.CUSTOMER_ID))

    for day in range(nb_days):
        # volumen con estacionalidad
        lam = max(0.0, float(customer_profile.mean_nb_tx_per_day)) * float(tx_mult[day])
        nb_tx = np.random.poisson(lam)

        if nb_tx > 0:
            for _ in range(nb_tx):
                time_tx = int(np.random.normal(86400 / 2, 20000))
                if 0 < time_tx < 86400:
                    amount = np.random.normal(customer_profile.mean_amount, customer_profile.std_amount)
                    if amount < 0:
                        amount = np.random.uniform(0, customer_profile.mean_amount * 2)

                    # aplica multiplicador y ruido estacional
                    mult = float(amount_mult[day])
                    noise = 1.0 + float(np.random.normal(0.0, amount_noise[day])) if amount_noise[day] > 0 else 1.0
                    amount = float(np.round(amount * mult * noise, 2))

                    if len(customer_profile.available_terminals) > 0:
                        terminal_id = random.choice(customer_profile.available_terminals)
                        customer_transactions.append([
                            time_tx + day * 86400, day, customer_profile.CUSTOMER_ID, terminal_id, amount
                        ])

    customer_transactions = pd.DataFrame(customer_transactions, columns=[
        'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT'
    ])
    if len(customer_transactions) > 0:
        customer_transactions['TX_DATETIME'] = pd.to_datetime(
            customer_transactions["TX_TIME_SECONDS"], unit='s', origin=start_date
        )
        customer_transactions = customer_transactions[[
            'TX_DATETIME','CUSTOMER_ID','TERMINAL_ID','TX_AMOUNT','TX_TIME_SECONDS','TX_TIME_DAYS'
        ]]
    return customer_transactions


def make_seasonality_maps(
    start_date="2025-01-01", nb_days=365*2,
    tx_mult_by_month={11: 1.20, 12: 1.35},         # +20% en nov, +35% en dic (volumen)
    amount_mult_by_month={11: 1.10, 12: 1.25},     # +10% en nov, +25% en dic (monto)
    amount_std_extra_by_month={11: 0.02, 12: 0.05} # ruido extra (0..0.05 ~= 5%) en dic
):
    import pandas as pd, numpy as np
    dates = pd.date_range(pd.to_datetime(start_date), periods=nb_days, freq="D")
    months = dates.month.values
    tx_mult = np.ones(nb_days, dtype="float32")
    amount_mult = np.ones(nb_days, dtype="float32")
    amount_noise = np.zeros(nb_days, dtype="float32")
    for m, v in tx_mult_by_month.items():       tx_mult[months==m] = v
    for m, v in amount_mult_by_month.items():   amount_mult[months==m] = v
    for m, v in amount_std_extra_by_month.items(): amount_noise[months==m] = v
    return tx_mult, amount_mult, amount_noise


# Generación completa del dataset
def generate_dataset(n_customers=10000, n_terminals=1000000, nb_days=90, start_date="2025-01-01", r=5):
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1
    )

    tx_mult, amount_mult, amount_noise = make_seasonality_maps(start_date=start_date, nb_days=nb_days)

    def _gen(g):
        cp = g.iloc[0]
        return generate_transactions_table_seasonal(
            cp, start_date=start_date, nb_days=nb_days,
            tx_mult=tx_mult, amount_mult=amount_mult, amount_noise=amount_noise
        )
    

    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID', group_keys=False).apply(_gen).reset_index(drop=True)

    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    transactions_df.reset_index(inplace=True)
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)
    return customer_profiles_table, terminal_profiles_table, transactions_df

def _prepare_for_storage_simple(df):
    """
    Deja un dataset compacto para modelado y particionado.
    - Quita IDs y timestamps redundantes.
    - NO guarda TX_FRAUD_SCENARIO.
    """
    df = df.copy()

    # columnas a eliminar si existen
    drop_cols = [
        'TRANSACTION_ID', 'CUSTOMER_ID', 'TERMINAL_ID',
        'TX_DATETIME', 'TX_DATE', 'TX_FRAUD_SCENARIO',
        'available_terminals'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

    # columnas a conservar (intersección por seguridad)
    keep_cols = [
        # particionado / tiempo numérico
        'TX_YEAR', 'TX_MONTH', 'TX_DAY', 'TX_TIME_DAYS', 'TX_TIME_SECONDS',
        # variables numéricas principales
        'TX_AMOUNT',
        'x_customer_id', 'y_customer_id', 'mean_amount', 'std_amount', 'mean_nb_tx_per_day',
        'x_terminal_id', 'y_terminal_id',
        # etiqueta
        'TX_FRAUD'
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    # dtypes ligeros
    cast_map = {
        'TX_AMOUNT': 'float32',
        'TX_TIME_DAYS': 'int32',
        'TX_TIME_SECONDS': 'int32',
        'TX_YEAR': 'int16',
        'TX_MONTH': 'int8',
        'TX_DAY': 'int8',
        'x_customer_id': 'float32', 'y_customer_id': 'float32',
        'mean_amount': 'float32', 'std_amount': 'float32', 'mean_nb_tx_per_day': 'float32',
        'x_terminal_id': 'float32', 'y_terminal_id': 'float32',
        'TX_FRAUD': 'int8'
    }
    for c, t in cast_map.items():
        if c in df.columns:
            df[c] = df[c].astype(t, copy=False)

    return df

def generate_and_save(
    n_customers=100_000, n_terminals=100_000,
    start_date="2025-01-01", nb_days=365*2, r=8,
    out_base="./data/fraud_stream_parquet",
    schedule=None,  # lista de escenarios (ver arriba)
    chunk_days=30   # generar/aplicar/guardar por mes
):
    from math import ceil

    
    print("Generando perfiles y transacciones base...")
    cust, term, tx = generate_dataset(
        n_customers=n_customers,
        n_terminals=n_terminals,
        nb_days=nb_days,
        start_date=start_date,
        r=r
    )


    print(f"Transacciones generadas: {len(tx)}")
    
    full = combine_profiles(tx, cust, term, start_date=start_date)

  
    if schedule is not None:
        full = apply_fraud_schedule(full, cust, term, schedule)

    full = _prepare_for_storage_simple(full)

    print("\nEstructura del DataFrame que se guardará:")
    print("Columnas:", full.columns.tolist())
    print(full.head())

   # transacciones_por_mes = full.groupby('TX_MONTH').size()
    # Mostrar el resultado
    #print("Transacciones por mes:")
    #print(transacciones_por_mes)

    print("\nEstructura del DataFrame que se guardará:")
    print("Columnas:", full.columns.tolist())
    print(full.head())

    # Mostrar la cantidad de fraudes por día
    fraudes_por_dia = full[full['TX_FRAUD'] == 1].groupby('TX_MONTH').size()
    print("\nCantidad de fraudes por día:")
    print(fraudes_por_dia)

    print("\nCantidad de NO fraudes por día:")
    print(full[full['TX_FRAUD'] == 0].groupby('TX_MONTH').size())

    write_parquet_by_chunks(full, out_base, freq="MS", compression="snappy")

    # Limpieza
    del cust, term, tx, full
    gc.collect()

def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    
    
    transactions_df['TX_FRAUD']=0
    transactions_df['TX_FRAUD_SCENARIO']=0
    
    # Scenario 1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD']=1
    transactions_df.loc[transactions_df.TX_AMOUNT>220, 'TX_FRAUD_SCENARIO']=1
    nb_frauds_scenario_1=transactions_df.TX_FRAUD.sum()
    print("Number of frauds from scenario 1: "+str(nb_frauds_scenario_1))
    
    # Scenario 2
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_terminals = terminal_profiles_table.TERMINAL_ID.sample(n=2, random_state=day)
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+28) & 
                                                    (transactions_df.TERMINAL_ID.isin(compromised_terminals))]
                            
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD']=1
        transactions_df.loc[compromised_transactions.index,'TX_FRAUD_SCENARIO']=2
    
    nb_frauds_scenario_2=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_1
    print("Number of frauds from scenario 2: "+str(nb_frauds_scenario_2))
    
    # Scenario 3
    for day in range(transactions_df.TX_TIME_DAYS.max()):
        
        compromised_customers = customer_profiles_table.CUSTOMER_ID.sample(n=3, random_state=day).values
        
        compromised_transactions=transactions_df[(transactions_df.TX_TIME_DAYS>=day) & 
                                                    (transactions_df.TX_TIME_DAYS<day+14) & 
                                                    (transactions_df.CUSTOMER_ID.isin(compromised_customers))]
        
        nb_compromised_transactions=len(compromised_transactions)
        
        
        random.seed(day)
        index_fauds = random.sample(list(compromised_transactions.index.values),k=int(nb_compromised_transactions/3))
        
        transactions_df.loc[index_fauds,'TX_AMOUNT']=transactions_df.loc[index_fauds,'TX_AMOUNT']*5
        transactions_df.loc[index_fauds,'TX_FRAUD']=1
        transactions_df.loc[index_fauds,'TX_FRAUD_SCENARIO']=3
        
                             
    nb_frauds_scenario_3=transactions_df.TX_FRAUD.sum()-nb_frauds_scenario_2-nb_frauds_scenario_1
    print("Number of frauds from scenario 3: "+str(nb_frauds_scenario_3))
    
    return transactions_df          


def save_dataset(transactions_df, output_dir="./Simuladores/Output/Fraud Detection Handbook/"):
    
    #return transactions_df
    start_date = datetime.datetime.strptime("2025-01-01", "%Y-%m-%d")
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')
        date = start_date + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d") + '.pkl'
        transactions_day.to_pickle(os.path.join(output_dir, filename_output), protocol=4)


if __name__ == "__main__":
    customer_profiles_table, terminal_profiles_table, transactions_df = generate_dataset(
        n_customers=5000, n_terminals=10000, nb_days=183, start_date="2025-01-01", r=5
    )
    save_dataset(transactions_df)
    print(transactions_df)
    print("Dataset generado y guardado correctamente.")
