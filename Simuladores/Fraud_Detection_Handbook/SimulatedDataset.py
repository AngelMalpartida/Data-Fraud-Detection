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

    # (Opcional) recorta a ~15 numéricas finales si lo deseas:
    # features = ["TX_AMOUNT","TX_TIME_SECONDS","TX_TIME_DAYS",
    #             "x_customer_id","y_customer_id","mean_amount","std_amount","mean_nb_tx_per_day",
    #             "x_terminal_id","y_terminal_id"]
    # df = df[["TX_DATETIME","CUSTOMER_ID","TERMINAL_ID"] + features + ["TX_FRAUD","TX_FRAUD_SCENARIO"]]

    return df

def add_fraud_scenario_1(df, amount_threshold=220.0, start_day=None, end_day=None):
    """
    Regla simple por monto. Aplica solo entre start_day y end_day (inclusive).
    """
    df = df.copy()
    mask_range = pd.Series(True, index=df.index)
    if start_day is not None: mask_range &= df["TX_TIME_DAYS"] >= start_day
    if end_day   is not None: mask_range &= df["TX_TIME_DAYS"] <= end_day

    mask = mask_range & (df["TX_AMOUNT"] > amount_threshold)
    df.loc[mask, "TX_FRAUD"] = 1
    df.loc[mask, "TX_FRAUD_SCENARIO"] = 1
    return df

def add_fraud_scenario_2(df, terminal_profiles_table, n_per_day=2, window_days=28, start_day=None, end_day=None, seed=0):
    """
    Terminals comprometidos por 'ventanas' deslizantes. Selección estable por día (semilla).
    """
    df = df.copy()
    max_day = int(df["TX_TIME_DAYS"].max())
    start = 0 if start_day is None else int(start_day)
    end   = max_day if end_day is None else int(end_day)

    rng = np.random.RandomState(seed)
    term_ids = terminal_profiles_table["TERMINAL_ID"].values

    for day in range(start, end+1):
        rng_day = np.random.RandomState(day)  # reproducible por día
        compromised = rng_day.choice(term_ids, size=min(n_per_day, len(term_ids)), replace=False)
        mask = (df["TX_TIME_DAYS"]>=day) & (df["TX_TIME_DAYS"]<day+window_days) & (df["TERMINAL_ID"].isin(compromised))
        df.loc[mask, "TX_FRAUD"] = 1
        df.loc[mask, "TX_FRAUD_SCENARIO"] = 2
    return df

def add_fraud_scenario_3(df, customer_profiles_table, n_customers_per_day=3, window_days=14,
                          amp_factor=5.0, frac_to_flip=1/3, start_day=None, end_day=None, seed=0):
    """
    Clientes comprometidos: amplifica montos y etiqueta una fracción como fraude.
    """
    df = df.copy()
    max_day = int(df["TX_TIME_DAYS"].max())
    start = 0 if start_day is None else int(start_day)
    end   = max_day if end_day is None else int(end_day)

    cust_ids = customer_profiles_table["CUSTOMER_ID"].values

    for day in range(start, end+1):
        rng_day = random.Random(day + seed)
        compromised = rng_day.sample(list(cust_ids), k=min(n_customers_per_day, len(cust_ids)))

        mask = (df["TX_TIME_DAYS"]>=day) & (df["TX_TIME_DAYS"]<day+window_days) & (df["CUSTOMER_ID"].isin(compromised))
        idxs = df.index[mask]
        if len(idxs) == 0: 
            continue

        # Amplifica montos
        df.loc[idxs, "TX_AMOUNT"] = (df.loc[idxs, "TX_AMOUNT"] * amp_factor).astype("float32")

        # Etiqueta una fracción como fraude
        k = max(1, int(len(idxs) * frac_to_flip))
        flip_idxs = rng_day.sample(list(idxs), k=k)
        df.loc[flip_idxs, "TX_FRAUD"] = 1
        df.loc[flip_idxs, "TX_FRAUD_SCENARIO"] = 3
    return df


def apply_fraud_schedule(df, customer_profiles_table, terminal_profiles_table, schedule):
    """
    schedule: lista de dicts, p.ej.:
    [
      {"scenario":1, "start_day":0,   "end_day":59,  "params":{"amount_threshold":220}},
      {"scenario":2, "start_day":60,  "end_day":119, "params":{"n_per_day":2, "window_days":28}},
      {"scenario":3, "start_day":120, "end_day":179, "params":{"n_customers_per_day":3, "window_days":14, "amp_factor":5}}
    ]
    """
    out = df.copy()
    for step in schedule:
        sc = step["scenario"]
        sd = step.get("start_day", None); ed = step.get("end_day", None)
        params = step.get("params", {}) or {}
        if sc == 1:
            out = add_fraud_scenario_1(out, start_day=sd, end_day=ed, **params)
        elif sc == 2:
            out = add_fraud_scenario_2(out, terminal_profiles_table, start_day=sd, end_day=ed, **params)
        elif sc == 3:
            out = add_fraud_scenario_3(out, customer_profiles_table, start_day=sd, end_day=ed, **params)
        else:
            raise ValueError(f"Escenario desconocido: {sc}")
    return out


def write_parquet_partitioned(df, base_path, compression="snappy", partition_cols=("TX_YEAR","TX_MONTH")):
    os.makedirs(base_path, exist_ok=True)
    # Partición por año/mes (y si quieres día) → lectura selectiva y archivos más pequeños
    df.to_parquet(
        base_path,
        engine="pyarrow",
        compression=compression,
        partition_cols=list(partition_cols),
        index=False
    )

def write_parquet_by_chunks(df, base_path, freq="MS", compression="snappy"):
    """
    Alternativa: cortar por mes y escribir uno a uno para bajar RAM pico.
    freq="": Month Start.
    """
    os.makedirs(base_path, exist_ok=True)

    df = df.sort_values("TX_DATETIME")
    # Asegúrate de NO guardar columnas pesadas como 'available_terminals'
    if "available_terminals" in df.columns:
        df = df.drop(columns=["available_terminals"])

    # Agrupa por año (freq 'YS' = Year Start)
    for _, g in df.groupby(pd.Grouper(key="TX_DATETIME", freq="YS")):
        if g.empty:
            continue
        y = int(g["TX_YEAR"].iloc[0])
        out_dir = os.path.join(base_path, f"TX_YEAR={y}")
        os.makedirs(out_dir, exist_ok=True)

        fn = os.path.join(out_dir, f"transactions_{y}.parquet")
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

# Generación completa del dataset
def generate_dataset(n_customers=10000, n_terminals=1000000, nb_days=90, start_date="2025-01-01", r=5):
    print(f"Generando perfiles de {n_customers} clientes...")
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    print("Perfiles de clientes generados.")
    print(f"Generando perfiles de {n_terminals} terminales...")
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    print("Perfiles de terminales generados.")
    print("Calculando terminales disponibles para cada cliente dentro del radio especificado...")
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1
    )

    print("Terminales disponibles calculados.")
    print(f"Generando transacciones para {n_customers} clientes durante {nb_days} días...")

    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(x.iloc[0], nb_days=nb_days)
    ).reset_index(drop=True)

    print("Ordenando y asignando IDs a las transacciones...")

    print("Transacciones ordenadas y listas.")

    print("Generación del dataset completada.")
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    transactions_df.reset_index(inplace=True)
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)
    return customer_profiles_table, terminal_profiles_table, transactions_df


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
    # Nota: para 10–20M filas puede convenir generar por bloques desde la función
    # (si lo necesitas, te paso versión "streaming" que itera por semanas/meses).

    print(f"Transacciones generadas: {len(tx)}")
    
    full = combine_profiles(tx, cust, term, start_date=start_date)

    # 3) Aplicar fraudes por calendario (opcional)
    if schedule is not None:
        full = apply_fraud_schedule(full, cust, term, schedule)

    # 4) Guardar en Parquet particionado (por mes)
    # Opción A: escribir todo a la vez (requiere RAM acorde)
    # write_parquet_partitioned(full, out_base, partition_cols=("TX_YEAR","TX_MONTH"))

    print("\nEstructura del DataFrame que se guardará:")
    print("Columnas:", full.columns.tolist())
    print("Primeras 5 filas:")
    print(full[['CUSTOMER_ID','TX_FRAUD_SCENARIO','TX_DATE', 'TX_DATETIME', 'TX_TIME_SECONDS', 'TX_TIME_DAYS', 'mean_nb_tx_per_day']].head())
    print(full['TX_FRAUD_SCENARIO'].unique())
    # Opción B: escribir por chunks mensuales (recomendado para 10–20M)
    write_parquet_by_chunks(full, out_base, freq="MS", compression="snappy")

    # Limpieza
    del cust, term, tx, full
    gc.collect()

def add_frauds(customer_profiles_table, terminal_profiles_table, transactions_df):
    
    # By default, all transactions are genuine
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


# Guardar el dataset
def save_dataset(transactions_df, output_dir="./Simuladores/Output/Fraud Detection Handbook/"):
    
    #return transactions_df
    start_date = datetime.datetime.strptime("2025-01-01", "%Y-%m-%d")
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')
        date = start_date + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d") + '.pkl'
        transactions_day.to_pickle(os.path.join(output_dir, filename_output), protocol=4)

# Ejecución principal
if __name__ == "__main__":
    customer_profiles_table, terminal_profiles_table, transactions_df = generate_dataset(
        n_customers=5000, n_terminals=10000, nb_days=183, start_date="2025-01-01", r=5
    )
    save_dataset(transactions_df)
    print(transactions_df)
    print("Dataset generado y guardado correctamente.")
