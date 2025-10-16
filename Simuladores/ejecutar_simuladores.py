# Script para ejecutar funciones específicas de los simuladores
import subprocess
import sys
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
  # Año 1
  {"scenario":1, "start_day":0,   "end_day":89,  "params":{"amount_threshold":220}},
  {"scenario":2, "start_day":90,  "end_day":179, "params":{"n_per_day":2, "window_days":28}},
  {"scenario":3, "start_day":180, "end_day":269, "params":{"n_customers_per_day":3, "window_days":14, "amp_factor":5, "frac_to_flip":1/3}},
  # overlap S1+S2
  {"scenario":1, "start_day":270, "end_day":359, "params":{"amount_threshold":200}},  # más agresivo
  {"scenario":2, "start_day":270, "end_day":359, "params":{"n_per_day":3, "window_days":21}},  # más denso
  # Año 2 (reapariciones)
  {"scenario":3, "start_day":360, "end_day":449, "params":{"n_customers_per_day":4, "window_days":14, "amp_factor":5}},
  {"scenario":1, "start_day":450, "end_day":539, "params":{"amount_threshold":230}},  # umbral más alto = menos fraudes
  # overlap S2+S3
  {"scenario":2, "start_day":540, "end_day":629, "params":{"n_per_day":2, "window_days":28}},
  {"scenario":3, "start_day":540, "end_day":629, "params":{"n_customers_per_day":2, "window_days":10, "amp_factor":4}},
  # fase baja
  {"scenario":1, "start_day":630, "end_day":729, "params":{"amount_threshold":260}},  # muy pocos S1
]


def crear_handbook_dataset():
    generate_and_save(
    n_customers=2000, n_terminals=2000,
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
    #ejecutar_sparkov_via_script(
      #  n_customers=200,
      #  output_dir='.Simuladores/Output/Sparkov/',
      #  start_date='01-01-2025',
      #  end_date='12-31-2025'
   # )
