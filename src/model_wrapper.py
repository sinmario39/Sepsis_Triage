import joblib
import pandas as pd
from preprocessing import prepare_patient


class MLModelWrapper:

    """
    Wrapper per l'integrazione dei modelli di ML all'interno dell'agente.

    Gestisce il caricamento dei modelli addestrati e fornisce un'interfaccia unica per ottenere le predizioni.

    Il wrapper nasconde i dettagli di implementazione del modello:
    l'agente riceve solamente la probabilità di sepsi e la classificazione della condizione clinica
    senza dover conoscere il tipo di algoritmo utilizzato.
    """

    def __init__(self, sepsis_path, macro_path):
        self.model_sepsis = joblib.load(sepsis_path)
        self.model_macro = joblib.load(macro_path)

    # Esegue la predizione sui dati clinici di un paziente
    def predict(self, patient_data):

        X = prepare_patient(patient_data)

        prob_sepsis = self.model_sepsis.predict_proba(X)[0][1]
        macro_pred = self.model_macro.predict(X)[0]

        return prob_sepsis, macro_pred