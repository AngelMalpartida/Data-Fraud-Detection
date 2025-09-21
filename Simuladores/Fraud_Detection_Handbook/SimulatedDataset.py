# Adaptación completa del notebook SimulatedDataset.ipynb a un script Python

import os
import datetime
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('darkgrid', {'axes.facecolor': '0.9'})

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
def generate_transactions_table(customer_profile, start_date="2018-04-01", nb_days=10):
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
def generate_dataset(n_customers=10000, n_terminals=1000000, nb_days=90, start_date="2018-04-01", r=5):
    customer_profiles_table = generate_customer_profiles_table(n_customers, random_state=0)
    terminal_profiles_table = generate_terminal_profiles_table(n_terminals, random_state=1)
    x_y_terminals = terminal_profiles_table[['x_terminal_id', 'y_terminal_id']].values.astype(float)
    customer_profiles_table['available_terminals'] = customer_profiles_table.apply(
        lambda x: get_list_terminals_within_radius(x, x_y_terminals=x_y_terminals, r=r), axis=1
    )
    transactions_df = customer_profiles_table.groupby('CUSTOMER_ID').apply(
        lambda x: generate_transactions_table(x.iloc[0], nb_days=nb_days)
    ).reset_index(drop=True)
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    transactions_df.reset_index(inplace=True)
    transactions_df.rename(columns={'index': 'TRANSACTION_ID'}, inplace=True)
    return customer_profiles_table, terminal_profiles_table, transactions_df

# Guardar el dataset
def save_dataset(transactions_df, output_dir="./Simuladores/Output/Fraud Detection Handbook/"):
    
    return transactions_df
    '''start_date = datetime.datetime.strptime("2018-04-01", "%Y-%m-%d")
    for day in range(transactions_df.TX_TIME_DAYS.max() + 1):
        transactions_day = transactions_df[transactions_df.TX_TIME_DAYS == day].sort_values('TX_TIME_SECONDS')
        date = start_date + datetime.timedelta(days=day)
        filename_output = date.strftime("%Y-%m-%d") + '.pkl'
        transactions_day.to_pickle(os.path.join(output_dir, filename_output), protocol=4)'''

# Ejecución principal
if __name__ == "__main__":
    customer_profiles_table, terminal_profiles_table, transactions_df = generate_dataset(
        n_customers=5000, n_terminals=10000, nb_days=183, start_date="2018-04-01", r=5
    )
    save_dataset(transactions_df)
    print(transactions_df)
    print("Dataset generado y guardado correctamente.")
