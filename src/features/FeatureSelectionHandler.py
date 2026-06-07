#!/usr/bin/env python3
# pylint: disable=C0103
"""
Handeling Feature Selection
"""
import os
import joblib
import optuna
import pandas as pd
from optuna.samplers import NSGAIISampler, RandomSampler

from ..modeling.OptunaHandler import OptunaHandler

class FeatureSelectionHandler:
    """
    This class implements feature selection using Optuna optimization framework.
    The implementation is inspired/adapted from https://towardsdatascience.com/feature-selection-with-optuna-0ddf3e0f7d8c/

    Parameters:

    - model (string): The predictive model to evaluate; 
                        this should be any object that implements fit() and predict() methods.
    - features (list of str): A list containing the names of all possible features 
                                that can be selected for the model.
    - x (DataFrame): The complete set of feature data (pandas DataFrame) 
                        from which subsets will be selected for training the model.
    - y (Series): The target variable associated with the X data (pandas Series).
    - penalty (float, optional): A factor used to penalize the objective function 
                                    based on the number of features used.
    """

    def __init__(self,
                 model,
                 features,
                 x,
                 y,
                 penalty=0.0005, 
                 label="feature_selection"):

        self.model = model.lower()
        self.features = features
        self.x = x
        self.y = y
        self.penalty = penalty
        self.label = label

    def get_cv_score(self,
                    trial,
                    n_trials=10,
                    n_splits=3,
                    sampler=RandomSampler(seed=42)):
        """
        Manages HP Optimization in Feature Selection. Returns loss of feature selection. 
        
        Parameters:
        
        - trial
        - n_trials (int): Number of trials for HP Optimization
        - n_splits (int): Number of splits for Crossvalidation
        - sampler (optuna.sampler): Sampler that creates the HP samples
        """

        pinned_features = []
        searchable_features = self.features

        # Let Optuna choose only for searchable features
        selected = {name: trial.suggest_categorical(name, [True, False])
                    for name in searchable_features}

        # Always include pinned features
        selected.update({name: trial.suggest_categorical(name, [True])
                    for name in pinned_features})

         # Selected feature names
        selected_feature_names = [name for name, include in selected.items() if include]

        # Adds a penalty for the amount of features used
        n_used = len(selected_feature_names)
        total_penalty = n_used * self.penalty

        # Start HP-Tuning and Feature Selection from OptunaHandler
        handler = OptunaHandler(self.x[selected_feature_names], self.y,
                                self.model, logging_active=False)
        handler.tune_model(n_trials,n_splits, sampler, device= 'cpu'
                           if self.model!='mlp' else 'cuda')
        trial_df = handler.model_df
        loss = -trial_df.loc[0,'MCC']

        # Add the penalty to the loss
        loss += total_penalty
        return loss

    def start_selection(self,
                        save_dir,
                        hp_trials=10,
                        hp_splits=3,
                        sampler= NSGAIISampler(seed=42,population_size=50),
                        n_trials=1000,
                        n_jobs=-1):
        """
        Starts Feature Selection
        
        Parameters:
        
        - save_dir (string): Directory to safe the resulting feature set and pkl file.
        - hp_trials (int): Number of trials for Hyperparameter Tuning 
                            for each Feature Selection Trial
        - hp_splits (int): Number of splits for Crossvalidation for HP Tuning
        - sampler (optuna.sampler): Sampler that creates features selection samples
        - n_trials (int): Number of trials for Feature Selection
        - n_jobs (int): Number of CPU cores to work in parallel (all = -1)        
        """
        study = optuna.create_study(direction="minimize",sampler=sampler)
        default_features = {ft: True for ft in self.x.columns}
        study.enqueue_trial(default_features)
        study.optimize(lambda trial: self.get_cv_score(trial=trial,
                                                       n_trials=hp_trials,
                                                       n_splits=hp_splits),
                        n_trials=n_trials, n_jobs=n_jobs)

        trial_df = study.trials_dataframe().reset_index(drop=True)
        trial_df = trial_df.rename({"value": "MCC Loss"}, axis=1)
        trial_df = trial_df.sort_values("MCC Loss", ascending=True).reset_index(drop=True)
        best_trial = trial_df.iloc[[0]]
        best_columns = best_trial.columns[best_trial.any()].tolist()
        best_columns = [col.replace("params_","")
                        for col in best_columns if col.startswith("params_")]
        trial_df['n_used_features']=trial_df[[col
                                              for col in trial_df.columns
                                              if col.startswith('param')]].apply(func=sum,axis=1)
        trial_df['MCC'] = -trial_df['MCC Loss']+(self.penalty * trial_df['n_used_features'])

        model_dir = f"{save_dir}/{self.model.upper()}"

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        trial_df.to_csv(f"{model_dir}/{self.label}_df.csv", index=False)
        joblib.dump(study, f"{model_dir}/{self.label}_study.pkl")

        return best_columns
