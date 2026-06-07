"""Evaluation of the Ensemble."""

import os

from dotenv import load_dotenv, find_dotenv
from .DataEnsemble import DataEnsemble

def main():
    """
    Function to execute the Ensemble train + Evaluation
    """
    load_dotenv(find_dotenv())
    data_dir = os.getenv("FEATURE_DIR")
    model_dir = os.getenv("MODEL_RESULTS")
    fes = [
           "paper-free"
           ]
    for fe in fes:
        print(fe)
        data_ensemble = DataEnsemble(feature_set=fe,
                                   path_to_data=data_dir,
                                   path_to_models=model_dir)
        data_ensemble.evaluate_ensemble(n_predictions=2)


if __name__ == "__main__":
    main()
