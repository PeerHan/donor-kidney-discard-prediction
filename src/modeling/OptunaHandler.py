#!/usr/bin/env python3
# pylint: disable=C0103
"""
Handeling Optuna
"""
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import optuna

from .optuna_toolkit import * # pylint: disable=W0401, W0614

class OptunaHandler:
    """
    Handling class for easy to use hyperparameter training for certain models.
    Currently the following models are available:
        - DT: Decisiontree
        - MLP: Multi Layer Perceptron
        - RF: RandomForest
        - xGB: extreme Gradient Boost
        - LR: Logistic Regression

    The handler trains a modell for n trials with a choosen sampler (TPE as default).
    The input data is divided in 5 folds for a 5-CV 
    where the validation fold is used to calculate the validation MCC.
    Each trial has one MCC which is the mean of the 5 MCC scores.

    The trial dataframe can be saved for further analysis or visualization purposes.
    The study object can also be stored as pickle file.
    """

    def __init__(self, x, y, algorithm, logging_active=True):
        """
        Parameters:
            - x: Training data which should be split in Train and Validation
            Data should be in a Machine Learning suitable format
            - y: Target Column - Data should be in a Machine Learning suitable format
            - algorithm: Used algorithm which should be one of the following: 'dt, mlp, rf, xgb'
        """
        self.logging_active = logging_active
        algorithm = algorithm.lower().strip()
        objectiv_map = {
            "dt" : dt_objective,
            "mlp" : mlp_objective,
            "rf" : rf_objective,
            "xgb" : xgb_objective,
            "lr" : lr_objective
        }
        assert algorithm in ["dt", "mlp", "rf", "xgb", "lr"], \
            "Algorithm should be: DT, MLP, RF or XGB"
        self.x_orig = x
        self.y_orig = y
        self.algorithm = algorithm
        self.study = None
        self.objective_function = objectiv_map[algorithm]
        self.model_df = None

    def log_trials(self, trials, x, y):
        """
        log trials
        """

        df = x.copy()

        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame(df, columns=[f"feature_{i}" for i in range(df.shape[1])])


        if isinstance(y, pd.Series):
            target_name = y.name
        else:
            y = pd.Series(y, name="target")
            target_name = y.name

        df[target_name] = y


        if self.logging_active:
            mfl = MlFlowLogger(self.algorithm)
            for trial in trials:
                mfl.start_run("optuna_trials")
                mfl.log_dataset(df, "basic", y.name, "training")
                mfl.log_params(trial.params)
                mfl.log_metrics({"MCC": trial.values[0]})
                mfl.end_run()

    def tune_model( # pylint: disable=R0913, R0914, R0917
            self,
            n_trials,
            n_splits=5,
            sampler=optuna.samplers.TPESampler(seed=42, multivariate=True),
            save_data=None,
            save_study=None,
            random_state=42,
            **kwargs
            ):
        """
        Start the model tuning - Warning: Could take some time for big trial size
        
        Parameters:
            - n_trials: Amount of trials (amount of models which are fitted, max is 100)
            - n_splits: Splits for cv - default: 5
            - sampler: Define which sampling strategy for the hyperparameter search is used
                - Doc: https://optuna.readthedocs.io/en/stable/reference/samplers/index.html
                - Default is the TPESampler 
                which can also be parameterized (here the default parameters are used)
            - save_data: Define a path with a csv file (e.g. /home/name/folder/example.csv)
            to save the trial as dataframe
            - save_study: Define a path with a pkl file (e.g. /home/name/folder/study.pkl) 
            to save the study as pickle.
            - random_state: State for datasplitting, should be the same for comparison reasons
            - device: Device parameter for MLP models on cuda
        """
        assert 1 <= n_trials <= 300, "n_trials should be between 1 and 300"

        if not "weight_tag" in kwargs: # pylint: disable=C0117
            kwargs["weight_tag"] = "Tmp"

        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(lambda trial: self.objective_function(trial,
                                                             self.x_orig,
                                                             self.y_orig,
                                                             n_splits=n_splits,
                                                             random_state=random_state,
                                                             device=kwargs["device"],
                                                             weight_tag=kwargs["weight_tag"]),
                                                             n_trials=n_trials)

        self.log_trials(study.trials, self.x_orig, self.y_orig)

        if save_study:
            joblib.dump(study, save_study)

        trial_df = study.trials_dataframe().reset_index(drop=True)
        trial_df = trial_df.rename({"value" : "MCC"}, axis=1)
        trial_df = trial_df.sort_values("MCC", ascending=False).reset_index(drop=True)
        self.model_df = trial_df

        if save_data:
            trial_df.to_csv(save_data, index=False)

    def load_study(self, save_path):
        """
        Load a study as pkl to use the optuna visualizations
        """
        study = joblib.load(save_path)
        return study

    def get_model(self):
        """
        Returns the unfitted model with the best parameters.
        If mlp is used a dictionary will be returned because the study contains parameters 
        which are model related but are not part of the model:
        - model: Pytorch model 
        - init_lr: Initial learning rate
        - batch_size: batch_size
        - weight_decay: weight_decay term for the optimizer
        - sched_patience: Patience for the learning rate scheduler
        - es_patience: Early Stopper Patience
        - max_epochs: Amount of maximum epochs which should be used for training
        """
        model_map = {
            "dt" : create_dt_from_result,
            "mlp" : create_mlp_from_result,
            "rf" : create_rf_from_result,
            "xgb" : create_xgb_from_result,
            "lr" : create_lr_from_result
        }

        model_funct = model_map[self.algorithm]

        if isinstance(self.y_orig, pd.Series):
            unique_classes = self.y_orig.unique()
        else:
            unique_classes = np.unique(self.y_orig)

        model = model_funct(self.model_df,
                            input_feats=self.x_orig.shape[1],
                            output_feats=len(unique_classes))
        return model

    def test_evaluation(self, x_test, y_test, path_to_folder, # pylint: disable=R0913, R0914, R0917
                        iterations=30, logging_active=True, **kwargs):
        """
        Method to Evaluate a trained (fitted) ML Algorithm on a extern Testset
        The model is trained with n different random states (can be configured in iterations)
        to sample a test-metric distribution for a model with a specific HP set.

        Parameters:
        - x_test, y_test: Test Features and Test Target
        - path_to_folder: Path to a Folder where the following files are saved:
            - path_to_folder/model_cm.png (Confusion Matrix for the test set)
            - path_to_folder/model.pkl (joblib dump or weights of the model)
            - path_to_folder/model_predictions.csv (Predictions with Probabilities of best model)
            - path_to_folder/model_metrics.csv
            (MCC and F1 Scores for every model with rand state)

        
        """
        device = kwargs["device"] if "device" in kwargs else "cpu"
        if not "weight_tag" in kwargs: # pylint: disable=C0117
            kwargs["weight_tag"] = "Tmp"
        res = self.get_model()

        train_test_map = {
            "dt" : fit_and_eval_sklearn,
            "mlp" : fit_and_eval_mlp,
            "rf" : fit_and_eval_sklearn,
            "xgb" : fit_and_eval_sklearn,
            "lr" : fit_and_eval_sklearn
            }

        train_test_function = train_test_map[self.algorithm]
        f1_test, mcc_test, _, _, cm_test = train_test_function(res,
                                                                self.x_orig,
                                                                self.y_orig,
                                                                x_test,
                                                                y_test,
                                                                path_to_folder,
                                                                iterations,
                                                                model=self.algorithm.upper(),
                                                                device=device,
                                                                weight_tag=kwargs["weight_tag"],
                                                                logging_active=logging_active)

        plt.figure(figsize=(8, 8), dpi=250)
        sns.set_style("darkgrid")
        sns.heatmap(cm_test,
                    cmap="viridis",
                    annot=True,
                    fmt=".0f",
                    xticklabels=["Nicht Transplantiert", "Transplantiert"],
                    yticklabels=["Nicht Transplantiert", "Transplantiert"],
                    annot_kws={"fontsize":30})
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel("Vorhersage", fontsize=20)
        plt.ylabel("Echtes Label", fontsize=20)
        plt.title(f"{self.algorithm} Performance auf dem Testdatensatz\n" +
                  f"F1 Score: {f1_test:.2} | Normed MCC: {((1 + mcc_test) / 2):.2}")
        plt.savefig(f"{path_to_folder}/{self.algorithm}_cm.png", bbox_inches="tight", dpi=250)
