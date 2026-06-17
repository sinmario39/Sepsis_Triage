import argparse
import os
import sys
import runpy

import pandas as pd
import random

# Importa i moduli da /src
def import_src_to_path():
    project_root = os.path.dirname(os.path.abspath(__file__)) # Prende la posizione del file
    src_path = os.path.join(project_root, "src") # Costruisce il path verso /src
    if src_path not in sys.path:
        sys.path.insert(0, src_path) # Aggiunge /src alla lista di path dove Python prende i moduli

# Esegue uno script Python dentro src come se fosse lanciato direttamente.
def run_level_script(script_name: str):

    project_root = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_root, "src", script_name)
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Non trovo {script_path}")
    runpy.run_path(script_path, run_name="__main__") # Questo file viene eseguito come se fosse stato lanciato con python

def parse_args():
    parser = argparse.ArgumentParser(
        description="ML Sepsis Project - Runner"
    )

    parser.add_argument(
        "--eda",
        action="store_true",
        help="Esegue l'analisi esplorativa dei dati (eda.py)"
    )

    parser.add_argument(
        "--level",
        choices=["1", "2", "all"],
        default="all",
        help="Quale livello eseguire (default: all)"
    )

    parser.add_argument(
        "--agent",
        action="store_true",
        help="Esegue il sistema intelligente di supporto alla diagnosi"
    )

    parser.add_argument(
        "--mode",
        choices=["eda", "train1", "train2", "trainall", "test"],
        help="Scelta del modello da eseguire"
    )
    return parser.parse_args()

def load_random_patient():

    df = pd.read_csv("data/snapshot.csv")

    FEATURES = [
        "Age",
        "Temp",
        "HR",
        "Resp",
        "O2Sat",
        "WBC",
        "Glucose",
        "Creatinine",
        "Lactate",
        "SBP",
        "DBP",
        "MAP",
        "Platelets",
        "BUN"
    ]

    # prende una riga casuale di un paziente e converte in un dictionary
    patient = df.sample(n=1).iloc[0]
    patient_data = patient[FEATURES].to_dict()

    return patient_data

def main():
    args = parse_args() # Legge gli argomenti della linea di comando
    import_src_to_path() # Aggiunge src al path

    # Esecuzione condizionale
    if args.mode == "eda":
        print("\n### Running EDA (eda.py) ###\n")
        run_level_script("eda.py")

    if args.mode == "train1" or args.mode == "trainall":
        print("\n### Running LEVEL 1 (train_level1.py) ###\n")
        run_level_script("train_level1.py")

    if args.mode == "train2" or args.mode == "trainall":
        print("\n### Running LEVEL 2 (train_level2.py) ###\n")
        run_level_script("train_level2.py")

    if args.mode == "test":
        from agent import MedicalAgent
        from model_wrapper import MLModelWrapper

        print("\n### Running AGENT (agent.py) ###\n")
        # Caricamento modelli
        model = MLModelWrapper("models/sepsis_DecisionTree(depth=5).pkl","models/macro_DecisionTree(depth=5).pkl")
        # Creazione agente
        agent = MedicalAgent(model)

        # Carica i dati di un paziente scelto casualmente
        patient_data = load_random_patient()

        # Valutazione
        result = agent.evaluate(patient_data)

        # Stampa dati del paziente
        print("\n=== PATIENT DATA ===\n")

        for k, v in patient_data.items():
            if pd.isna(v):
                print(f"{k}: non disponibile")
            else:
                print(f"{k}:  {v}")

        print("\n=== RULE BASED EXPLANATION WITH EVIDENCES ===\n")
        print(result["explanation"])

        # Stampa risultati della diagnosi
        print("=== DECISION ENGINE ===")

        print("\n")
        print("Decisione finale:", result["diagnosis"])
        print("Score decisionale:", round(result["confidence"], 2))
        print("Motivazione:", result["reason"])


if __name__ == "__main__":
    main()