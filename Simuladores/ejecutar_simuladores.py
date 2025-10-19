# Script para ejecutar funciones específicas de los simuladores
import subprocess
import sys
import pandas as pd
import os
import pathlib
config_file_path = pathlib.Path(__file__).parent / 'SparkovDataGeneration' / 'profiles' / 'main_config.json'


# La ejecucion se debe hacer con python -m Simuladores.ejecutar_simuladores
#from .ADV_O.ADVO.generator import Generator
from Fraud_Detection_Handbook.SimulatedDataset import generate_dataset,add_frauds,generate_and_save
#from .SparkovDataGeneration.datagen_customer import main as datagen_customers
#from .SparkovDataGeneration.datagen_transaction import main as datagen_transactions
#from .SparkovDataGeneration.datagen_transaction import valid_date

# Configuración de directorios de salida
output_dir_adv_o = './Simuladores/Output/ADV-O/'
output_dir_fraud_detection = './Simuladores/Output/Fraud Detection Handbook/'



def instalar_dependencias():

    print("Instalando dependencias para ADV-O...")
    
    # Verificar el directorio actual
    print(f"Directorio actual: {os.getcwd()}")
    

    print("Instalando dependencias para ADV-O...")
    subprocess.run(['pip', 'install', '-r', 'Simuladores/ADV_O/requirements.txt'], check=True)

    #print("Instalando dependencias para Fraud Detection Handbook...")
    #subprocess.run(['pip', 'install', '-r', 'Simuladores/Fraud_Detection_Handbook/requirements.txt'], check=True)



schedule_1 = [
  # ======================
  # PRETRAIN (Meses 1–4): S1 + S2 simultáneos
  # ======================
  {"scenario": 1, "start_day":   0, "end_day": 119, "params": {"amount_threshold": 220}},          # S1
  {"scenario": 2, "start_day":   0, "end_day": 119, "params": {"n_per_day": 2, "window_days": 21}},# S2

  # ======================
  # Bimestres (S1 → S2 → S3) hasta 24 meses
  # ======================

  # Bloque 1 (May–Jun 2025): S1
  {"scenario": 1, "start_day": 120, "end_day": 180, "params": {"amount_threshold": 225}},

  # Bloque 2 (Jul–Ago 2025): S2
  {"scenario": 2, "start_day": 181, "end_day": 242, "params": {"n_per_day": 2, "window_days": 28}},

  # Bloque 3 (Sep–Oct 2025): S3
  {"scenario": 3, "start_day": 243, "end_day": 303, "params": {"n_customers_per_day": 3, "window_days": 14, "amp_factor": 5, "frac_to_flip": 1/3}},

  # Bloque 4 (Nov–Dic 2025, NAVIDAD): S1 suavizado
  {"scenario": 1, "start_day": 304, "end_day": 364, "params": {"amount_threshold": 250}},  # umbral más alto → menos S1

  # Bloque 5 (Ene–Feb 2026): S2
  {"scenario": 2, "start_day": 365, "end_day": 423, "params": {"n_per_day": 2, "window_days": 28}},

  # Bloque 6 (Mar–Abr 2026): S3
  {"scenario": 3, "start_day": 424, "end_day": 484, "params": {"n_customers_per_day": 3, "window_days": 14, "amp_factor": 5, "frac_to_flip": 1/3}},

  # Bloque 7 (May–Jun 2026): S1
  {"scenario": 1, "start_day": 485, "end_day": 545, "params": {"amount_threshold": 225}},

  # Bloque 8 (Jul–Ago 2026): S2
  {"scenario": 2, "start_day": 546, "end_day": 607, "params": {"n_per_day": 2, "window_days": 28}},

  # Bloque 9 (Sep–Oct 2026): S3
  {"scenario": 3, "start_day": 608, "end_day": 668, "params": {"n_customers_per_day": 3, "window_days": 14, "amp_factor": 5, "frac_to_flip": 1/3}},

  # Bloque 10 (Nov–Dic 2026, NAVIDAD): S1 suavizado
  {"scenario": 1, "start_day": 669, "end_day": 729, "params": {"amount_threshold": 250}}
]



def crear_handbook_dataset():
    generate_and_save(
    n_customers=1500, n_terminals=1500,
    start_date="2025-01-01", nb_days=365*2, r=8,
    out_base = os.path.join(os.environ['USERPROFILE'], 'Downloads', 'fraud_stream_parquet'),
    schedule=schedule_1,  # lista de escenarios (ver arriba)
    chunk_days=30   # generar/aplicar/guardar por mes
    )
   

# Ejecutar simulador ADV-O
def ejecutar_adv_o():
    print("Ejecutando simulador ADV-O...")

    # Inicializar Generator con los parámetros necesarios
    generator = Generator(n_jobs=1, radius=8)

    # Llamar al método generate con los argumentos adecuados
    transactions_df = generator.generate(
        filename='ADVO_df.csv', 
        nb_days_to_generate=180, 
        max_days_from_compromission=7, 
        n_terminals=1000, 
        n_customers=500, 
        compromission_probability=0.01,
        start_date="2025-01-01"
    )

    # Guardar el DataFrame generado
    if not os.path.exists(output_dir_adv_o):
        os.makedirs(output_dir_adv_o)
    transactions_df.to_csv(os.path.join(output_dir_adv_o, 'ADVO_df.csv'), index=False)

    print("Simulador ADV-O completado y datos guardados.")

# Ejecutar simulador Fraud Detection Handbook
def ejecutar_fraud_detection():
    print("Ejecutando simulador Fraud Detection Handbook...")
    customer_profiles_table, terminal_profiles_table, transactions_df = generate_dataset(
        n_customers=500, n_terminals=1000, nb_days=180, start_date="2025-01-01", r=8
    )

    transactions_df=add_frauds(customer_profiles_table,terminal_profiles_table,transactions_df)

    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(output_dir_fraud_detection):
        os.makedirs(output_dir_fraud_detection)

    # Imprimir la ruta donde se guardará el archivo
    print(f"Guardando transactions_df en: {os.path.join(output_dir_fraud_detection, 'Handbook_df.csv')}")

    # Guardar el DataFrame generado
    transactions_df.to_csv(os.path.join(output_dir_fraud_detection, 'Handbook_df.csv'), index=False)
    terminal_profiles_table.to_csv(os.path.join(output_dir_fraud_detection, 'terminal_profiles.csv'), index=False)
    customer_profiles_table.to_csv(os.path.join(output_dir_fraud_detection, 'customer_profile.csv'), index=False)



    print("Simulador Fraud Detection Handbook completado.")


# Ejecutar simulador Sparkov con parámetros simplificados
def ejecutar_sparkov_simple(n_customers, output_dir, start_date, end_date):
    print("Ejecutando simulador Sparkov con parámetros simplificados...")

    # Crear directorio de salida si no existe
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Rutas internas
    config_path = pathlib.Path('./Simuladores/SparkovDataGeneration/profiles/main_config.json')
    customers_out_file = pathlib.Path(output_dir) / 'customers.csv'
    transactions_out_file = pathlib.Path(output_dir) / 'transactions.csv'

    # Generar clientes
    datagen_customers(n_customers, 42, config_path, customers_out_file)

    # Generar transacciones
    datagen_transactions(
        customer_file=customers_out_file,
        profile_file=config_path,
        start_date=valid_date(start_date),
        end_date=valid_date(end_date),
        out_path=transactions_out_file
    )

    print(f"Simulador Sparkov completado. Transacciones guardadas en: {transactions_out_file}")

def ejecutar_sparkov_via_script(n_customers, output_dir, start_date, end_date):
    print("Ejecutando simulador Sparkov mediante script datagen.py...")

    # Construir el comando para ejecutar datagen.py
    command = [
        'python','-m', 'Simuladores.SparkovDataGeneration.datagen',
        '--config', str(config_file_path),
        '-n', str(n_customers),
        '-o', output_dir,
        start_date, end_date
    ]

    # Ejecutar el comando
    subprocess.run(command, check=True)

    print(f"Simulador Sparkov completado. Resultados guardados en: {output_dir}")

if __name__ == "__main__":
    # Ejecutar ambos simuladores
    #instalar_dependencias()
    #ejecutar_adv_o()
    #ejecutar_fraud_detection()
    crear_handbook_dataset()
    # Ejecutar simulador Sparkov
    #Para ejecutar en la linea de comandos
    #Se debe dirigir a su ruta y ejecutar:
    #python datagen.py -n 500 -o ../Output/SparkovDataGeneration 01-01-2025 06-30-2025
    
