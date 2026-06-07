#!/usr/bin/env python3
"""
Making Logging easier
"""

from datetime import datetime

import mlflow
from mlflow.data.pandas_dataset import PandasDataset # pylint: disable=W0611


class MlFlowLogger:
    """
    Class to build logging objects
    """

    def __init__(self, algo, tracking_uri="http://morgan:5000"):
        """
        Initialize the MLFlowLogger
        - algorithm: dt, mlp, rf, xgb, lr
        - stage: test, log (for final model), validate (for final validation)
        - input_table: spender, labor, op-explantation, diagnose, transplantation, alle
        """
        dt_iso = datetime.now().isoformat()
        self.algorithm = algo
        self.run = None
        assert self.algorithm.lower() in ["dt", "mlp", "rf", "xgb", "lr"]
        mlflow.set_tracking_uri(tracking_uri) # access UI via http://localhost:5555
        mlflow.set_experiment(str(dt_iso[:-7]) + "-" + self.algorithm)


    def start_run(self, stage):
        """
        Initialize the MLFlowLogger
        - stage: test, log (for final model), validate (for final validation)
        - input_table: spender, labor, op-explantation, diagnose, transplantation, alle
        """
        self.run = mlflow.start_run()
        mlflow.set_tag("model_target", "target") # time_series, target, nlp
        mlflow.set_tag("stage", stage)

    def log_dataset(self, data, name, target, context):
        """
        Dataset loggen
        - data: Pandas Datframe mit allen Daten (Train und Test, X und y)
        - name: "basic", "expert"
        - target: Zielvariable/Name von y-Spalte
        """
        dataset = mlflow.data.from_pandas(data, name=name, targets=target) # pylint: disable=E1101
        mlflow.log_input(dataset, context=context)

    def log_params(self, params):
        """
        Logging the Parameters of the model
        """
        mlflow.log_params(params)

    def log_metrics(self, metrics):
        """
            Logging the Metrics of the model
            - f1: F1-Score
            - mcc: Matthews Correlation Coeffficent
            - cm: Confusion Matrix
            - precision: Precision
            - recall: Recall
         """
        mlflow.log_metrics(metrics)


    def log_model(self, model):
        """
        Log Model
        """

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=self.algorithm,
            registered_model_name=self.algorithm
        )

    def end_run(self):
        """End the MLflow run."""
        mlflow.end_run()
