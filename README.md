# Comparative Evaluation of Machine Learning Models for Predicting Donor Kidney Discard
Despite the growing use of machine learning to predict donor kidney discard, reported results remain inconsistent due to heterogeneous datasets, variable preprocessing, and non-standardized evaluation strategies. This study provides a systematic and reproducible comparison of machine learning models for donor kidney discard prediction, emphasizing methodological rigor over model complexity. We trained five commonly used algorithms: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, and Multilayer Perceptron, along with an ensemble model, using data from 4,080 deceased donors in Germany. An unified benchmarking framework was implemented, incorporating standardized feature engineering, supervised feature selection, and Bayesian hyperparameter optimization. Model performance was evaluated in terms of discrimination (normed Matthews Correlation Coefficient, AUC, F1), calibration (Brier score), and explainability (SHAP). The ensemble model achieved the highest discrimination performance ($\text{MCC}_\text{Normed}$ = 0.76, AUC = 0.87, F1 = 0.90), while Logistic Regression, Random Forest, and the Multilayer Perceptron performed comparably and outperformed single Decision Trees. Calibration improved notably through Platt scaling for tree-based and neural network models. SHAP analysis consistently identified donor age and renal function markers as dominant predictors across models, demonstrating clinical plausibility and interpretability. Overall, our findings show that consistent data preprocessing, feature selection, and evaluation are more critical for predictive success than the specific choice of machine learning algorithm, providing a transparent and reproducible framework for improving donor kidney discard prediction and supporting clinical decision-making in transplantation.

## Citation
```
@article{schliephacke2026comparative,
  title={Comparative Evaluation of Machine Learning Models for Predicting Donor Kidney Discard},
  author={Schliephacke, Peer and Schult, Hannah and Mizera, Leon and W{\"u}rfel, Judith and Grieser, Gunter and Rahmel, Axel and Fischer-Fr{\"o}hlich, Carl-Ludwig and Jahn-Eimermacher, Antje},
  journal={arXiv preprint arXiv:2602.21876},
  year={2026}
}
```

# Getting Started

## Create Virtual Env
We use `Python 3.11.2` to create and install the venv with all requirements: 
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Feature Engineering 
```
python src/data/make_data.py
```
will run all preprocessing and feature engineering steps. The resulting datasets are saved in the FEATURE_DIR specified in the .env file.

## Feature Selection
After running the command:
```
python3 -m src.features.create_paper_feature_set --model dt -p "PATH_TO_ARTEFACT_FOLDER"
```
Two artifacts will be saved in a subfolder dedicated to the respective model (DT in this case). 
The subfolder contains a dataframe with metadata and a binarized Optuna study object. 
Additional arguments can be set, such as adjusting the number of trials (-t num).

The feature selection can also be executed consecutiv for all model families using:
```
bash src/features/feature_selection_all_models.sh
```
It should be noted that if no CUDA device is available, the MLP may take some time to run depending on the number of trials.

## Hyperparameter Tuning and Evaluation
Model hyperparameters are tuned on the training split using the feature subset identified during feature selection. 
Afterwards, the best model configuration is re-trained on the training split and evaluated on the unseen test split.
The evaluation involves a discriminative and explainability analysis:
```
bash src/modeling/train_cpu_models.sh
bash src/modeling/train_gpu_models.sh
```
All artifacts can be assessed under the artefact folder containing:
- The best model and performances/visualizations around it
- The predictions of all evaluation runs
- Explainability plots and data regarding SHAP
- Meta-data und the optuna study object (binary) for the hyperparameter tuning

## Ensemble
Analogous artifacts can be created for the ensemble model using:
```
python3 -m src.modeling.eval_de
```

## Post-Hoc Analysis, Calibration and visualization
Run the following command for all post-hoc analysis including calibration and the estimation of confidence intervals.
All corresponding files must be included in the respective folder.
```
bash src/post_hoc_analysis/run_all_analysis.sh
```

All resulting artifacts of the post-hoc analysis can be assessed under zenodo including all output files and figures: