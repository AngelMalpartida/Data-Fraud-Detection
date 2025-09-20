# import SVG
import pandas as pd
import numpy as np
import logging
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata
# from sdv.utils import get_random_subset
from sdv.single_table import CTGANSynthesizer

logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":

    dataset = load_csvs("Simuladores/")
    fraud_object = dataset['credit_dataset_2_transaccion']
    print(fraud_object.head())

    metada_fraud = Metadata.detect_from_dataframe(
        data=fraud_object, table_name="fraud_data")

    # Y opcionalmente si quieres indicar el tiempo:
    '''
    metada_fraud.update_column(
        column_name="TransactionDT",
        sdtype="datetime",
        datetime_format="%s",  # segundos desde el inicio
        tags={
            "time_index": True
        }
    )

    
    subsampled_data = get_random_subset(
        dataset,
        metada_fraud,
        num_sequences=100
    )'''

    synthesizer = CTGANSynthesizer(metada_fraud,
                                   enforce_rounding=False,
                                   epochs=500,
                                   cuda=True,
                                   verbose=True)

    synthesizer.fit(fraud_object)

    synthetic_data = synthesizer.sample(num_rows=10)

    print(synthetic_data.head())
