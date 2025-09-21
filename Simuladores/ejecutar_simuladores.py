# Script para ejecutar funciones específicas de los simuladores
import subprocess
import sys
import os

from .ADV_O.ADVO.generator import Generator
from Fraud_Detection_Handbook.SimulatedDataset import generate_dataset, save_dataset

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


# Ejecutar simulador ADV-O
def ejecutar_adv_o():
    print("Ejecutando simulador ADV-O...")

    # Inicializar Generator con los parámetros necesarios
    generator = Generator(n_jobs=1, radius=8)

    # Llamar al método generate con los argumentos adecuados
    transactions_df = generator.generate(
        filename='dataset_six_months.csv', 
        nb_days_to_generate=183, 
        max_days_from_compromission=7, 
        n_terminals=10000, 
        n_customers=5000, 
        compromission_probability=0.01
    )

    # Guardar el DataFrame generado
    if not os.path.exists(output_dir_adv_o):
        os.makedirs(output_dir_adv_o)
    transactions_df.to_csv(os.path.join(output_dir_adv_o, 'transactions_df.csv'), index=False)

    print("Simulador ADV-O completado y datos guardados.")

# Ejecutar simulador Fraud Detection Handbook
def ejecutar_fraud_detection():
    print("Ejecutando simulador Fraud Detection Handbook...")
    customer_profiles_table, terminal_profiles_table, transactions_df = generate_dataset(
        n_customers=5000, n_terminals=10000, nb_days=183, start_date="2018-04-01", r=5
    )

    # Verificar si la carpeta de salida existe, si no, crearla
    if not os.path.exists(output_dir_fraud_detection):
        os.makedirs(output_dir_fraud_detection)

    # Imprimir la ruta donde se guardará el archivo
    print(f"Guardando transactions_df en: {os.path.join(output_dir_fraud_detection, 'transactions_df.csv')}")

    # Guardar el DataFrame generado
    transactions_df.to_csv(os.path.join(output_dir_fraud_detection, 'transactions_df.csv'), index=False)

    print("Simulador Fraud Detection Handbook completado.")

if __name__ == "__main__":
    # Ejecutar ambos simuladores
    #instalar_dependencias()
    ejecutar_adv_o()
    ejecutar_fraud_detection()
