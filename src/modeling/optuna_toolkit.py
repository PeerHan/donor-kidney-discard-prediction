#!/usr/bin/env python3
"""
Functions for Optuna Tuning
"""

import numpy as np

from xgboost import XGBClassifier
from torch import nn
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler

from .machine_learning_toolkit import * # pylint: disable=W0401, W0614


def create_model(trial, input_feats, output_feats):
    """
    Create a Pytorch model based on a optuna trial with the
    following HPs:
    - Layers: 1-10
    - Hidden dim: 100-1500
    - Activity: ReLU, Leak, Mish, Swish, GELU, ELU
    - dropout prob: 0-0.5
    - batchnorm layer: True or False
    """

    n_layer = trial.suggest_int("n_layer", 2, 15, step=1)
    hidden_dim = trial.suggest_int("hidden_dim", 100, 1500)
    activation_function = trial.suggest_categorical("activation_function",
                         ["ReLU", "Leaky", "Mish", "Swish", "GELU", "ELU"])
    dropout_p = trial.suggest_float("dropout", 0., 0.5, step=0.05)
    batchnorm = bool(trial.suggest_int("batchnorm", 0, 1))

    function_map = {
        "ReLU" : nn.ReLU(),
        "Leaky" : nn.LeakyReLU(),
        "Mish" : nn.Mish(),
        "Swish" : nn.SiLU(),
        "GELU" : nn.GELU(),
        "ELU" : nn.ELU()
    }

    layers = []
    in_dim = input_feats

    for _ in range(n_layer):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))

        if batchnorm:
            layers.append(nn.BatchNorm1d(int(hidden_dim)))

        layers.append(function_map[activation_function])

        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))

        in_dim = int(hidden_dim)

    layers.append(nn.Linear(in_dim, output_feats))

    return nn.Sequential(*layers)

def mlp_objective(trial, x, y, n_splits=5, random_state=420, show_every=None, **kwargs): # pylint: disable=R0913, R0914, R0917
    """
    Objective to train a MLP involving the MLP Model HPs and other HPs:
    - weight_decay: 1e-10 - 1e-6 (log)
    - init learningrate: 1e-4 - 1e-1 (log)
    - max_epochs: 10 - 500
    - use_class_weights: Balanced Class weights (yes or no)
    """
    assert "device" in kwargs, \
        "keyword for device is missing e.g. handler.tune_model(..., device='cpu')"
    device = kwargs["device"].lower()
    assert device in ["cpu", "cuda"], "device should be cpu or cuda"

    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)
    batch_size = 512
    weight_decay = trial.suggest_float("weight_decay", high=1e-6, low=1e-10, log=True)
    init_lr = trial.suggest_float("init_lr", high=1e-1, low=1e-4, log=True)
    #patience = 5
    #es_patience = 5
    #es_delta = 0.0
    max_epochs = 1000
    use_class_weights = bool(trial.suggest_int("class_weights", 0, 1))

    input_feats = x.shape[1]
    output_feats = np.unique(y).shape[0]
    average_mcc = []
    for train_indices, val_indices in kf.split(x):
        torch.manual_seed(random_state)
        x_train = x.iloc[train_indices].reset_index(drop=True)
        y_train = y[train_indices].reset_index(drop=True)
        x_val = x.iloc[val_indices].reset_index(drop=True)
        y_val = y[val_indices].reset_index(drop=True)
        class_weights = compute_class_weight(class_weight="balanced",
                                             classes=np.unique(y_train),
                                             y=y_train)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)) \
            if use_class_weights else nn.CrossEntropyLoss()
        trainset = TorchDataSet(x_train, y_train)
        train_scaler = trainset.scaler
        valset = TorchDataSet(x_val, y_val, train_scaler)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
        model = create_model(trial, input_feats=input_feats, output_feats=output_feats)
        model.apply(initialize_weights)

        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        train_model(model=model,
                    optimizer=optimizer,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    criterion=criterion,
                    temp_weight_tag=kwargs["weight_tag"],
                    device=device,
                    max_epochs=max_epochs,
                    num_classes=output_feats,
                    show_every=show_every)
        _, mcc_val, _, _, _, _ = test_model(model, val_loader, device, num_classes=output_feats) # pylint: disable=W0632
        average_mcc.append(mcc_val)
    return np.mean(average_mcc)

def dt_objective(trial, x, y, n_splits=5, random_state=420, **kwargs): # pylint: disable=W0613
    """
    Objective to train a DT involving the DT HPs and other HPs:
    - max_depth: 1-50
    - min_samples_split: 2-20
    - min_samples_leaf: 1-20
    - max_features: sqrt, log2, or None
    - class_weight: balanced dataset or None
    """

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    model = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
        class_weight=class_weight,
    )
    x = StandardScaler().fit_transform(x)
    mcc_scorer = make_scorer(matthews_corrcoef)
    average_mcc = cross_val_score(model, x, y, cv=kf, scoring=mcc_scorer)

    return np.mean(average_mcc)


def rf_objective(trial, x, y, n_splits=5, random_state=420, **kwargs): # pylint: disable=W0613, R0914
    """
    Objective to train a RF involving the RF HPs and other HPs:
    - n_estimators: 10-200
    - max_depth: 1-50
    - min_samples_split: 2-20
    - min_samples_leaf: 1-20
    - max_features: sqrt, log2, or None
    - class_weight: balanced dataset or None
    """

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    n_estimators = trial.suggest_int("n_estimators", 10, 500)
    max_depth = trial.suggest_int("max_depth", 1, 50)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    x = StandardScaler().fit_transform(x)
    mcc_scorer = make_scorer(matthews_corrcoef)
    average_mcc = cross_val_score(model, x, y, cv=kf, scoring=mcc_scorer)

    return np.mean(average_mcc)

def lr_objective(trial, x, y, n_splits=5, random_state=420, **kwargs): # pylint: disable=W0613
    """
    Objective to train a RF involving the RF HPs and other HPs:
    - C: 0.01-10
    - solver: 'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'
    - class_weight: balanced dataset or None
    - l1_ration: percentage mixing between l1 and l2
    """

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    c = trial.suggest_float('C', 0.01, 100)
    class_weight = trial.suggest_categorical('class_weight', [None, 'balanced'])
    l1_ratio = trial.suggest_float('l1_ratio', 0, 1)

    model = LogisticRegression(
        C=c,
        solver='saga',
        class_weight=class_weight,
        l1_ratio=l1_ratio,
        penalty='elasticnet',
        max_iter=1000,
        n_jobs=-1,
        random_state=random_state
    )

    x = StandardScaler().fit_transform(x)
    mcc_scorer = make_scorer(matthews_corrcoef)
    average_mcc = cross_val_score(model, x, y, cv=kf, scoring=mcc_scorer)

    return np.mean(average_mcc)

def create_model_from_trial( # pylint: disable=R0913, R0917
        input_feats,
        n_layer,
        hidden_dim,
        batchnorm,
        activation_function,
        dropout_p,
        output_feats
        ):
    """
    Function to create a PyTorch Model in dependence of a optuna Trialdataframe
    """
    function_map = {
        "ReLU" : nn.ReLU(),
        "Leaky" : nn.LeakyReLU(),
        "Mish" : nn.Mish(),
        "Swish" : nn.SiLU(),
        "GELU" : nn.GELU(),
        "ELU" : nn.ELU()
    }

    layers = []
    in_dim = input_feats

    for _ in range(n_layer):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))

        if batchnorm:
            layers.append(nn.BatchNorm1d(int(hidden_dim)))

        layers.append(function_map[activation_function])

        if dropout_p > 0.0:
            layers.append(nn.Dropout(dropout_p))

        in_dim = int(hidden_dim)

    layers.append(nn.Linear(in_dim, output_feats))

    return nn.Sequential(*layers)

def create_mlp_from_result( # pylint: disable=W0102
        dataframe,
        tuned_feats=['params_activation_function',
                    'params_batchnorm', 'params_dropout',
                    'params_hidden_dim',
                    'params_init_lr', 'params_n_layer',
                    'params_weight_decay',
                    'params_class_weights'], 
        best_k=0,
        **kwargs
        ):
    """
    Create a MLP and all related parameters from an optuna study df.
    All Parameters are saved in a dictionary containing:
    - model: Pytorch model 
        - init_lr: Initial learning rate
        - batch_size: batch_size
        - weight_decay: weight_decay term for the optimizer
        - sched_patience: Patience for the learning rate scheduler
        - es_patience: Early Stopper Patience
        - max_epochs: Amount of maximum epochs which should be used for training
    """
    input_feats = kwargs["input_feats"]
    output_feats = kwargs["output_feats"]
    best_trial = dataframe.sort_values("MCC", ascending=False).\
        reset_index(drop=True).loc[best_k, tuned_feats].to_dict()
    best_trial = {name.replace("params_", "") : param for name, param in best_trial.items()}
    res_dict = {}
    model = create_model_from_trial(input_feats=input_feats,
                                    n_layer=best_trial["n_layer"],
                                    hidden_dim=best_trial["hidden_dim"],
                                    batchnorm=best_trial["batchnorm"],
                                    dropout_p=best_trial["dropout"],
                                    output_feats=output_feats,
                                    activation_function=best_trial["activation_function"])
    model.apply(initialize_weights)
    res_dict["model"] = model
    res_dict["init_lr"] = best_trial["init_lr"]
    #res_dict["batch_size"] = best_trial["batch_size"]
    res_dict["weight_decay"] = best_trial["weight_decay"]
    #res_dict["sched_patience"] = best_trial["sched_patience"]
    #res_dict["es_patience"] = best_trial["es_patience"]
    res_dict["max_epochs"] = 1000
    res_dict["class_weights"] = best_trial["class_weights"]
    return res_dict


def xgb_objective( # pylint: disable=R0914
        trial,
        x,
        y,
        n_splits=5,
        random_state=420,
        **kwargs # pylint: disable=W0613
    ):
    """
    Objective to train a XGBoost involving the MLP XGBoost HPs and other HPs:
    #- tree_method:             hist or approx
    #- learning_rate:           0.001-0.1 (log)
    #- max_depth:               3-17
    #- min_child_weight:        1-300
    #- reg_lambda:              0.001-25.0 (log)
    #- reg_alpha:               0.001-25.0 (log)
    #- colsample_bytree:        0.1-1.0
    #- subsample:               0.1-1.0
    #- early_stopping_rounds:   5-100
    #- max_leaf_nodes:          0-1000
    #- use_class_weights:       Balanced Class weights (yes or no)
    #- early_stopping_rounds:   100-1000
    """

    if isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
    if isinstance(y, np.ndarray):
        y = pd.Series(y)

    kf = KFold(n_splits=n_splits, random_state=random_state, shuffle=True)

    param = {
        'tree_method': trial.suggest_categorical(
            'tree_method', ['approx', 'hist']
            ),
        'reg_lambda': trial.suggest_float(
            'reg_lambda', 0.001, 25, log=True
            ),
        'reg_alpha': trial.suggest_float(
            'reg_alpha', 0.001, 25, log=True
            ),
        'colsample_bytree': trial.suggest_float(
            'colsample_bytree', 0.75, 1.0
            ),
        'subsample': trial.suggest_float(
            'subsample', 0.75, 1.0
            ),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 0.1, log=True
            ),
        'n_estimators': trial.suggest_int(
            'n_estimators', 100, 1500, step=50
        ),
        'max_depth': trial.suggest_int(
            'max_depth', 2, 15
            ),
        'min_child_weight': trial.suggest_int(
            'min_child_weight', 1, 20
            ),
        'early_stopping_rounds': trial.suggest_int(
            'early_stopping_rounds', 5, 100, step=5
            ),
        'n_jobs': -1,
        'random_state': random_state
    }
    use_class_weights = bool(trial.suggest_int("use_class_weights", 0, 1))
    average_mcc = []
    for train_indices, val_indices in kf.split(x):
        x_train = x.iloc[train_indices].reset_index(drop=True)
        y_train = y[train_indices].reset_index(drop=True)
        x_val = x.iloc[val_indices].reset_index(drop=True)
        y_val = y[val_indices].reset_index(drop=True)
        if use_class_weights:
            weight = y_train.value_counts()[0] / y_train.value_counts()[1]
            param["scale_pos_weight"] = weight
        model = XGBClassifier(**param)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        model.fit(x_train,y_train,eval_set=[(x_val,y_val)],verbose=False)
        preds = model.predict(x_val)
        mcc = matthews_corrcoef(y_val, preds)
        average_mcc.append(mcc)

    return np.mean(average_mcc)

def create_xgb_from_result( # pylint: disable=W0102
    dataframe,
    tuned_feats=['params_n_estimators', 'params_max_depth',
                 'params_min_child_weight', 'params_early_stopping_rounds',
                 'params_use_class_weights',
                 'params_tree_method',
                 'params_learning_rate', 'params_subsample', 'params_colsample_bytree',
                 'params_reg_alpha', 'params_reg_lambda'],
    n_jobs=-1,
    random_state=420,
    best_k=0,
    **kwargs # pylint: disable=W0613
    ):
    """
    Create a XGB and all related parameters from an optuna study df.
    All Parameters are saved in a dictionary containing:
    - model: XGBClassifier
        - tree_method:             hist or approx
        - learning_rate:           0.001-0.1 (log)
        - max_depth:               3-17
        - min_child_weight:        1-300
        - reg_lambda:              0.001-25.0 (log)
        - reg_alpha:               0.001-25.0 (log)
        - colsample_bytree:        0.1-1.0
        - subsample:               0.1-1.0
        - early_stopping_rounds:   5-100
        - max_leaf_nodes:          0-1000
        - use_class_weights:       Balanced Class weights (yes or no)
        - early_stopping_rounds:   100-1000
    """

    best_trial = dataframe.sort_values("MCC",ascending=False)\
    .reset_index(drop=True).loc[best_k, tuned_feats].to_dict()
    best_trial = {name.replace("params_", "") : param for name, param in best_trial.items()}

    res_dict={}
    model = XGBClassifier(
        n_estimators=best_trial["n_estimators"],
        max_depth=best_trial["max_depth"],
        min_child_weight=best_trial["min_child_weight"],
        early_stopping_rounds=best_trial['early_stopping_rounds'],
        tree_method=best_trial['tree_method'],
        learning_rate=best_trial['learning_rate'],
        subsample=best_trial['subsample'],
        colsample_bytree=best_trial['colsample_bytree'],
        reg_alpha=best_trial['reg_alpha'],
        reg_lambda=best_trial['reg_lambda'],
        n_jobs=n_jobs,
        random_state=random_state
    )

    res_dict["model"]=model
    res_dict['use_class_weights']=best_trial['use_class_weights']

    return res_dict

def create_dt_from_result( # pylint: disable=W0102
        dataframe,
        tuned_feats=['params_max_depth', 'params_min_samples_split',
                     'params_min_samples_leaf', 'params_max_features', 'params_class_weight'],
        random_state=420,
        best_k=0,
        **kwargs # pylint: disable=W0613
        ):
    """
    Create a DT and all related parameters from an optuna study df.
    All Parameters are saved in a dictionary containing:
    - model: DecisionTreeClassifier
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum number of samples required to split an internal node
        - min_samples_leaf: Minimum number of samples required to be at a leaf node
        - max_features: Number of features to consider when looking for the best split
        - class_weight: balanced dataset or None
    """

    best_trial = dataframe.sort_values("MCC", ascending=False)\
        .reset_index(drop=True).loc[best_k, tuned_feats].to_dict()
    best_trial = {name.replace("params_", "") : param for name, param in best_trial.items()}

    res_dict = {}
    model = DecisionTreeClassifier(
        max_depth=best_trial["max_depth"],
        min_samples_split=best_trial["min_samples_split"],
        min_samples_leaf=best_trial["min_samples_leaf"],
        max_features=best_trial["max_features"],
        class_weight=None if best_trial["class_weight"]\
            is np.nan else best_trial["class_weight"],
        random_state=random_state
    )

    res_dict["model"] = model
    res_dict["max_depth"] = best_trial["max_depth"]
    res_dict["min_samples_split"] = best_trial["min_samples_split"]
    res_dict["min_samples_leaf"] = best_trial["min_samples_leaf"]
    res_dict["max_features"] = best_trial["max_features"]
    res_dict["class_weight"] = None if best_trial["class_weight"]\
            is np.nan else best_trial["class_weight"]

    return res_dict


def create_rf_from_result( # pylint: disable=W0102
        dataframe,
        tuned_feats=['params_n_estimators', 'params_max_depth', 'params_min_samples_split',
                     'params_min_samples_leaf', 'params_max_features', 'params_class_weight'],
        random_state=420,
        best_k=0,
        **kwargs # pylint: disable=W0613
        ):
    """
    Create a DT and all related parameters from an optuna study df.
    All Parameters are saved in a dictionary containing:
    - model: DecisionTreeClassifier
        - n_estimators: The number of trees in the forest
        - max_depth: Maximum depth of the tree
        - min_samples_split: Minimum number of samples required to split an internal node
        - min_samples_leaf: Minimum number of samples required to be at a leaf node
        - max_features: Number of features to consider when looking for the best split
        - class_weight: balanced dataset or None
    """

    best_trial = dataframe.sort_values("MCC", ascending=False)\
        .reset_index(drop=True).loc[best_k, tuned_feats].to_dict()
    best_trial = {name.replace("params_", "") : param for name, param in best_trial.items()}

    res_dict = {}
    model = RandomForestClassifier(
        n_estimators=best_trial["n_estimators"],
        max_depth=best_trial["max_depth"],
        min_samples_split=best_trial["min_samples_split"],
        min_samples_leaf=best_trial["min_samples_leaf"],
        max_features=best_trial["max_features"],
        class_weight=None if best_trial["class_weight"]\
            is np.nan else best_trial["class_weight"],
        n_jobs=-1,
        random_state=random_state
    )

    res_dict["model"] = model
    res_dict["n_estimators"] = best_trial["n_estimators"]
    res_dict["max_depth"] = best_trial["max_depth"]
    res_dict["min_samples_split"] = best_trial["min_samples_split"]
    res_dict["min_samples_leaf"] = best_trial["min_samples_leaf"]
    res_dict["max_features"] = best_trial["max_features"]
    res_dict["class_weight"] = None if best_trial["class_weight"]\
            is np.nan else best_trial["class_weight"]

    return res_dict


def create_lr_from_result( # pylint: disable=W0102
        dataframe,
        tuned_feats=['params_C', 'params_class_weight', 'params_l1_ratio'],
        random_state=420,
        best_k=0,
        **kwargs # pylint: disable=W0613
        ):
    """
    Create a DT and all related parameters from an optuna study df.
    All Parameters are saved in a dictionary containing:
    - model: DecisionTreeClassifier
        - C: inverse of the regulation strength
        - solver: algo for optimizing the model
        - class_weight: balanced dataset or None
        - l1_ratio: percentage mixture of l1 and l2
    """

    best_trial = dataframe.sort_values("MCC", ascending=False)\
        .reset_index(drop=True).loc[best_k, tuned_feats].to_dict()
    best_trial = {name.replace("params_", "") : param for name, param in best_trial.items()}

    res_dict = {}
    model = LogisticRegression(
        C=best_trial["C"],
        solver='saga',
        class_weight=None if best_trial["class_weight"]\
            is np.nan else best_trial["class_weight"],
        l1_ratio=best_trial['l1_ratio'],
        penalty='elasticnet',
        max_iter=1000,
        random_state=random_state,
        n_jobs=-1
    )

    res_dict["model"] = model
    res_dict["C"] = best_trial["C"]
    res_dict["class_weight"] = best_trial["class_weight"]
    res_dict['l1_ratio'] = best_trial["l1_ratio"]
    return res_dict
