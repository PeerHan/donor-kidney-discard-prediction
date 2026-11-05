import os
import warnings
from dotenv import load_dotenv
import yaml

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer # pylint: disable=unused-import
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge
from feature_engine.selection import DropConstantFeatures, DropDuplicateFeatures

from utils import clear_csv_files

# Clean up console output
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
# Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version.
pd.set_option('future.no_silent_downcasting', True)

# Preprocessor Object
class DataPreprocessor:
    def __init__(self, target='Transplantationsstatus'):
        """Initialize preprocessor with target variable name."""
        self.target = target
        self.high_missing_ratio_cols = [] # Store columns for high missing ratio impute
        self.skewed_cols = [] # Store columns for SimpleImputer
        self.most_common_imp = None  # Store SimpleImputer for skewed columns
        self.int_cols = None
        self.scaler = None # Store StandardScaler for Imputing
        self.iterative_imp = None  # Store IterativeImputer for other columns
        self.dcf = None # Store DropConstantFeatures for test columns
        self.ddf = None # Store DropDuplicateFeatures for test columns
        self.dfc = None # Store DropFeaturesWithConstantValues for test columns
        self.train_values = {}

        load_dotenv(override=True)
        self.FEATURE_DIR = os.getenv("FEATURE_DIR")

    def _load_config(self, config_path):
        """Load YAML configuration file."""
        with open(config_path, 'r', encoding='UTF-8') as f:
            config = yaml.safe_load(f)
        return config

    def _get_path(self, file_name):
        """Get file path from environment variables."""
        return os.getenv(file_name)

    def _rearrange_config(self, config, target_file_name):
        """Move target file to end of configuration."""
        other_files = [file for file in config['excel_files'] if file['name'] != target_file_name]
        target_file = [file for file in config['excel_files'] if file['name'] == target_file_name]
        if not target_file:
            raise ValueError(
                f"File with name '{target_file_name}' not found in the configuration."
            )
        config["excel_files"] = other_files + target_file
        return config

    def move_column_to_last(self, df, column_name):
        """Move specified column to last position in DataFrame."""
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame.")
        columns = [col for col in df.columns if col != column_name]
        columns.append(column_name)
        return df[columns]
    
    def _prepare_saving(self, dataset_name, dev):
        """Prepare the directory for saving the dataset."""
        dataset_path = os.path.join(
            self.FEATURE_DIR,
            "dev",
            dataset_name
        ) if dev else os.path.join(self.FEATURE_DIR,dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        return dataset_path

    def _save_data(self, df, dataset_path, test_split):
        """Save the DataFrame to CSV files, splitting if necessary."""
        if test_split:
            df_train, df_test = self.split_by_ids(df)
            df_test.to_csv(os.path.join(dataset_path, 'test.csv'), index=False)
            df = df_train
        else:
            df_train = df
        df_train.to_csv(os.path.join(dataset_path, 'train.csv'), index=False)

    def split_by_ids(self, df):
        """Split DataFrame into train and test sets based on predefined IDs."""
        train_keepid = pd.read_csv(os.path.join(self.FEATURE_DIR,"train-test-split/train_ids.csv"))
        test_keepid = pd.read_csv(os.path.join(self.FEATURE_DIR,"train-test-split/test_ids.csv"))
        return df[df['SPM_ID'].isin(train_keepid['SPM_ID'])], df[df['SPM_ID'].isin(test_keepid['SPM_ID'])]

    def filter_organ(self, df, organs):
        """Filter DataFrame to include only specified organs."""
        df = df[df['dwh.Transplantat.Transplantat'].str.contains('|'.join(organs))]
        df = df.drop(columns='dwh.Transplantat.Transplantat')
        return df
    
    def generate_prefix(self, name):
        parts = name.split(' ')
        prefix = parts[0] + '.' + '_'.join(parts[1:])
        return prefix

    def filter_columns(self, df, config):
        columns_to_keep = ["SPM_ID", "Transplantationsstatus"]
        for file in config['excel_files']:
            file_name = file['name']
            print('Processing file:', file_name)
            for sheet in file['sheets']:
                cols_in_sheet = []
                if 'columns_to_keep' in sheet:
                    cols_in_sheet += sheet['columns_to_keep']
                    if "SPM_ID" in cols_in_sheet:
                            cols_in_sheet.remove("SPM_ID")
                if 'columns_to_drop' in sheet:
                    cols_in_sheet += [col for col in df.columns if col not in sheet['columns_to_drop'] and col.startswith(self.generate_prefix(sheet['name']))]
                if 'columns_to_combine' in sheet:
                    new_name = sheet['new_column_name'][0]
                    cols_in_sheet.append(new_name)
                if 'encoding_prefix' in sheet:
                    prefix = sheet['encoding_prefix'][0]
                    cols_in_sheet = [col for col in df.columns if col.startswith(prefix)]
                if 'columns_to_multiply' in sheet:
                    new_col_names = []
                    for col in sheet['columns_to_multiply']:
                        cols_in_sheet.remove(col)
                        for val in sheet['allowed_values']:
                            new_col_names.append(col + '_' + '_'.join(val.split(' ')))
                    cols_in_sheet += new_col_names
                if 'combine' in sheet:
                    for col in sheet['combine']:
                        if col['method'] == 'first':
                            for c in col['columns']:
                                new_col_name = c + "_first_value"
                                cols_in_sheet.remove(c)
                        if col['method'] == 'lowest':
                            for c in col['columns']:
                                new_col_name = c + "_min_value"
                                cols_in_sheet.remove(c)
                        cols_in_sheet.append(new_col_name)
                columns_to_keep += cols_in_sheet
        if "Rauchen_ICD_10_Code" in columns_to_keep:
            rauchen_cols = [col for col in df.columns if col.startswith("Rauchen_ICD_10_Code")]
            columns_to_keep += rauchen_cols
        if "Alkohol_ICD_10_Code" in columns_to_keep:
            alkohol_cols = [col for col in df.columns if col.startswith("Alkohol_ICD")]
            columns_to_keep += alkohol_cols
        columns_to_keep = [col_df for col_df in df.columns if any(col_df.startswith(col) for col in columns_to_keep)]

        # Remove columns based on dataset
        if config['new_dataset_name'] == 'exploration':
            columns_to_keep.remove('dwh.Monitoring.Diurese_Stunden')
            columns_to_keep.remove('dwh.Monitoring.Diurese_Menge')
            columns_to_keep.remove('Hepatitis_C_Diagnose')
            columns_to_keep = [col for col in columns_to_keep if not (col.startswith("Rauchen_ICD_10_Code") or col.startswith("Alkohol_ICD"))]
            # Remove columns that belong in domain
            domain_config = self._load_config('src/data/domain_config.yml')
            labor_entry = next(item for item in domain_config['excel_files'] if item['name'] == 'Labor_20_24')
            kcbu_sheet = next(sheet for sheet in labor_entry['sheets'] if sheet['name'] == 'dwh Labor KCBU')
            for col in kcbu_sheet['combine']:
                if col['method'] == 'first':
                    for c in col['columns']:
                        new_col_name = c + "_first_value"
                        if new_col_name in columns_to_keep:
                            columns_to_keep.remove(new_col_name)
                if col['method'] == 'lowest':
                    for c in col['columns']:
                        new_col_name = c + "_min_value"
                        if new_col_name in columns_to_keep:
                            columns_to_keep.remove(new_col_name)
        return df[columns_to_keep]
     
    def scale(self, df, scale_config):
        for col in scale_config:
            if col['method'] == 'µmol/L_to_mg/dL':
                df[col['name']] = df[col['name']] * 0.011312
        return df

    def _create_cerebrovascular_death(self, df, create_config):
        df[create_config["name"]] = np.where(
            df.Tod_ICD_10_Code.str.startswith("I6"), 1, 0
        )
        return df

    def _create_circulatory_death(self, df, create_config):
        df[create_config["name"]] = np.where(
            df.Tod_ICD_10_Code.str.startswith("I") & (~df.Zerebrovaskulaerer_Tod),
            1,
            0,
        )
        df.drop(columns="Tod_ICD_10_Code", inplace=True)
        return df

    def _create_duration_min(self, df, create_config):
        # calculate duration in minutes
        df[create_config['name']] = (df[create_config['end_col']]
                                      - df[create_config['start_col']]).dt.total_seconds() / 60

        # remove start and end column
        df.drop(
            columns=[create_config["start_col"], create_config["end_col"]], inplace=True
        )

        return df

    def _create_dauer_alkoholkonsum(self, df, create_config):
        # If patient doesn't drink alcohol (no start date), duration is 0
        df['Alkohol_Diagnose_Zeitpunkt_bis'] = df['Alkohol_Diagnose_Zeitpunkt_bis'].fillna(
            df['stationaere_Aufnahme_Zeitpunkt'])
        df[create_config['name']] = np.where(
            df['Alkohol_Diagnose_Zeitpunkt_von'].isna(),
            0,
            # Otherwise calculate duration between start and end/admission
            (df['Alkohol_Diagnose_Zeitpunkt_bis'] -
             df['Alkohol_Diagnose_Zeitpunkt_von']).dt.days
        )
        return df

    def _create_letzter_alkoholkonsum(self, df, create_config):
        # Calculate days since last drink
        df['days_since_last_drink'] = np.where(
            df['Alkohol_Diagnose_Zeitpunkt_von'].isna(),
            np.nan,  # No alcohol consumption
            np.where(
                df['Alkohol_Diagnose_Zeitpunkt_von'].notna() & df['Alkohol_Diagnose_Zeitpunkt_bis'].isna(),
                0,  # Still consuming
                (df['stationaere_Aufnahme_Zeitpunkt'] - df['Alkohol_Diagnose_Zeitpunkt_bis']).dt.days
            )
        )

        # Compute bins from the data
        max_days = df['days_since_last_drink'].max(skipna=True)
        bins = [-np.inf, 0] + list(np.linspace(1, max_days, num=4)) + [np.inf]
        labels = range(len(bins) - 1, 0, -1)

        # Print the bin ranges
        print("Bin ranges for 'days_since_last_drink':", bins)

        # Bin the days since last drink
        df[create_config['name']] = pd.cut(df['days_since_last_drink'], bins=bins, labels=labels, right=False)

        # Ensure the category 0 is included in the categories
        df[create_config['name']] = df[create_config['name']].cat.add_categories([0]).fillna(0).astype(int)

        # Drop the temporary column
        df.drop(columns=['days_since_last_drink'], inplace=True)

        return df

    def _create_diabetes_alter(self, df, create_config):
        df[create_config['name']] = np.where(
            df['Diabetes_Diagnose_Zeitpunkt_von'].notna(),
            df['dwh.Spender.Spender_Alter'] - ((df['stationaere_Aufnahme_Zeitpunkt']
                                               - df['Diabetes_Diagnose_Zeitpunkt_von']).dt.days / 365),
            0
        )
        return df

    def _create_diabetes_dauer(self, df, create_config):
        # If there was never any diabetes (no start date), set duration to 0
        df[create_config['name']] = np.where(
            df['Diabetes_Diagnose_Zeitpunkt_von'].isna(),
            0,
            # Otherwise calculate duration between start and end dates
            (df['Diabetes_Diagnose_Zeitpunkt_bis'].fillna(df['Diabetes_Diagnose_Zeitpunkt_von'])
             - df['Diabetes_Diagnose_Zeitpunkt_von']).dt.days
        )
        return df

    def _create_chemo_dauer(self, df, create_config):
        # If there was never any chemotherapy (no start date), set duration to 0
        df[create_config['name']] = np.where(
            df['Chemotherapie_Zeitpunkt_von'].isna(),
            0,
            # Otherwise calculate duration between start and end dates
            (df['Chemotherapie_Zeitpunkt_bis'].fillna(df['stationaere_Aufnahme_Zeitpunkt'])
            - df['Chemotherapie_Zeitpunkt_von']).dt.days
        )
        return df

    def _create_radio_dauer(self, df, create_config):
        # If there was never any radiation therapy (no start date), set duration to 0
        df[create_config['name']] = np.where(
            df['Strahlentherapie_Zeitpunkt_von'].isna(),
            0,
            # Otherwise calculate duration between start and end dates
            (df['Strahlentherapie_Zeitpunkt_bis'].fillna(df['stationaere_Aufnahme_Zeitpunkt'])
            - df['Strahlentherapie_Zeitpunkt_von']).dt.days
        )
        return df

    def _create_followup_dauer(self, df, create_config):
        # If there was never any follow-up care (no start date), set duration to 0
        df[create_config['name']] = np.where(
            df['Nachsorge_Zeitpunkt_von'].isna(),
            0,
            # Otherwise calculate duration between start and end dates
            (df['Nachsorge_Zeitpunkt_bis'].fillna(df['stationaere_Aufnahme_Zeitpunkt'])
            - df['Nachsorge_Zeitpunkt_von']).dt.days
        )
        return df

    def _create_norm_stand(self, df, create_config):
        if create_config['method2'] == 'stand':
            df[create_config['name']] = df[create_config['column_to_stand']]/df[create_config['stand_with']]
        elif create_config['method2'] == 'norm':
            df[create_config['name']] = ((df[create_config['zaehler']] / df[create_config['nenner']])
                                          * create_config['factor'] / df[create_config['stand_with']])
            df.drop(columns=[create_config['zaehler'], create_config['nenner']], inplace=True)
        return df

    def _create_reanimation_dauer(self, df, create_config):
        df[create_config['name']] = np.where(
            df['Reanimation_Diagnose_Zeitpunkt_von'].isna() | df['Reanimation_Diagnose_Zeitpunkt_bis'].isna(),
            np.nan,
            # Otherwise calculate duration between start and end dates in minutes
            (df['Reanimation_Diagnose_Zeitpunkt_bis']
            - df['Reanimation_Diagnose_Zeitpunkt_von']).dt.total_seconds() / 60
        )
        df.drop(columns=['Reanimation_Diagnose_Zeitpunkt_von', 'Reanimation_Diagnose_Zeitpunkt_bis'], inplace=True)
        return df

    def create_features(self, df, create_configs):
        """Create features based on the specified configuration."""
        method_map = {
            'create_cerebrovascular_death': self._create_cerebrovascular_death,
            'create_circulatory_death': self._create_circulatory_death,
            'alkdauer': self._create_dauer_alkoholkonsum,
            'alkletzt': self._create_letzter_alkoholkonsum,
            'diabetesalter': self._create_diabetes_alter,
            'diabetesdauer': self._create_diabetes_dauer,
            'create_duration_min': self._create_duration_min,
            'chemodauer': self._create_chemo_dauer,
            'radiodauer': self._create_radio_dauer,
            'followupdauer': self._create_followup_dauer,
            'norm_stand': self._create_norm_stand,
            'reanimationdauer': self._create_reanimation_dauer,
        }

        for creation in create_configs:
            method = method_map.get(creation['method'])
            if method:
                df = method(df, creation)
        return df

    def _encode_binary_emptyornot(self, df, column):
        df[column] = df[column].notna().astype(int)
        return df

    def _encode_binary(self, df, column, positive="Ja", negative="Nein"):
        """Encode column as binary based on positive and negative values. Positive and negative can be lists."""
        df[column] = df[column].apply(
            lambda x: 1 if (x in positive if isinstance(positive, list) else x == positive)
            else (0 if (x in negative if isinstance(negative, list) else x == negative)
            else np.nan)
        )
        return df

    def _encode_threshold(self, df, column, value):
        """Encode column as binary based on threshold value."""
        df[column] = df[column].apply(lambda x: 1 if x >= value else 0)
        return df

    def _encode_one_hot(self, df, column):
        """One-hot encode column, handling duplicate IDs if present."""
        df_one_hot = pd.get_dummies(df[column], prefix=column, dummy_na=False).astype(int)
        df_combined = pd.concat([df, df_one_hot], axis=1)
        df_combined = df_combined.drop(columns=[column])
        return df_combined

    def _encode_binary_custom(self, df, column, positive_value, new_column_prefix=None):
        """Create two binary columns for positive and negative values."""
        if new_column_prefix is None:
            new_column_prefix = column
        positive_col = f"{new_column_prefix}_Positive"
        negative_col = f"{new_column_prefix}_Negative"

        df[positive_col] = (df[column] == positive_value).astype(int)
        df[negative_col] = (df[column] != positive_value).astype(int)

        df = df.drop(columns=[column])
        return df

    def _encode_binary_with_multiple_positive_values(self,
                                                     df,
                                                     column,
                                                     positive_values,
                                                     negative_value):
        positive_col = f'{column}_Ja'
        negative_col = f'{column}_Nein'

        df[positive_col] = df[column].isin(positive_values).astype(int)
        df[negative_col] = (df[column] == negative_value).astype(int)

        df = df.drop(columns=[column])

        return df

    def _encode_icd(self, df, column, min_percentage=0.01):
        """One-hot encode ICD codes, excluding rare codes based on percentage threshold."""
        value_counts = df[column].value_counts()
        total_rows = len(df)

        percentages = value_counts / total_rows

        code_mapping = {
            code: code if percentage >= min_percentage else np.nan
            for code, percentage in percentages.items()
        }

        df[column] = df[column].map(code_mapping)
        df_one_hot = pd.get_dummies(df[column], prefix=column, dummy_na=False).astype(int)
        df_combined = pd.concat([df, df_one_hot], axis=1)
        df_combined = df_combined.drop(columns=[column])

        return df_combined

    def _encode_posnegmiss(self, df, column):
        """Encode 'posnegmiss' categories into two binary columns."""
        positive_col = f"{column}_Positive"
        negative_col = f"{column}_Negative"

        # Define the mapping for the categories
        positive_values = [1]
        negative_values = [0]

        # Create binary columns
        df[positive_col] = df[column].apply(lambda x: 1 if x in positive_values else 0)
        df[negative_col] = df[column].apply(lambda x: 1 if x in negative_values else 0)

        # Drop the original column
        df = df.drop(columns=[column])
        return df

    def encode(self, df, encoding_config):
        """Apply various encoding methods based on configuration."""

        method_map = {
            'binary': self._encode_binary,
            'binary_emptyornot': self._encode_binary_emptyornot,
            'threshold': self._encode_threshold,
            'binary_custom': self._encode_binary_custom,
            'one_hot': self._encode_one_hot,
            'icd': self._encode_icd,
            'encode_binary_with_multiple_positive_values': self._encode_binary_with_multiple_positive_values,
            'posnegmiss': self._encode_posnegmiss,
        }


        for method, columns in encoding_config.items():
            encode_function = method_map.get(method)
            if encode_function:
                for col_cfg in columns:
                    if method in ['binary', 'binary_custom']:
                        df = encode_function(df, col_cfg['name'], col_cfg['positive'], col_cfg['negative'])
                    elif method == 'threshold':
                        df = encode_function(df, col_cfg['name'], col_cfg['value'])
                    elif method == 'binary_custom':
                        df = encode_function(df,
                                             col_cfg['name'],
                                             col_cfg['positive_value'],
                                             col_cfg['new_column_prefix'])
                    elif method == 'encode_binary_with_multiple_positive_values':
                        df = encode_function(df,
                                             col_cfg['name'],
                                             col_cfg['positive_values'],
                                             col_cfg['negative_value'])
                    else:
                        df = encode_function(df, col_cfg)

        return df
    
    def drop_columns(self, df, drop_config):
        """Drop specified columns from DataFrame."""
        drop_config = [col for col in drop_config if col in df.columns]
        return df.drop(columns=drop_config)

    def rename_columns(self, df, rename_config):
        """Rename DataFrame columns according to configuration."""
        return df.rename(columns=rename_config)
    
    def _apply_optional_steps(self, df, config):
        """Apply optional steps like scaling, imputing, creating features, encoding, dropping, and renaming."""
        steps = {
            'scale': self.scale,
            'create': self.create_features,
            'encode': self.encode,
            'drop': self.drop_columns,
            'rename': self.rename_columns
        }
        for step, func in steps.items():
            if step in config:
                df = func(df, config[step])
            print("--", step)
        return df
    
    def sample_normal_from_column(self, col, size):
        """Function for dynamic imputation using normal distribution inferred
        from column values"""
        # Calculate min, max, mu and sigma from column (ignore NaN)
        lower_bound = col.mean() - 1.96 * col.std()
        upper_bound = col.mean() + 1.96 * col.std()

        # Filter
        column = col[(col >= lower_bound) & (col <= upper_bound)]

        data_min = column.min()
        data_max = column.max()
        mu = column.mean()
        sigma = column.std()
        # Number of missing values
        #size = column.isna().sum()

        sampled_values = []
        while len(sampled_values) < size:
            # Sample from Normaldistribution
            np.random.seed(42)
            samples = np.random.normal(mu, sigma, size)
            # Filter values in interval
            valid_samples = samples[(samples >= data_min) & (samples <= data_max)]
            sampled_values.extend(valid_samples)

        return np.array(sampled_values[:size])

    def impute(self, df, impute_config, encode_config, missing_threshold=0.7, is_train=True):
        # Fill ECMO with NLP results
        if "dwh.Monitoring.ECMO_Therapie" in df.columns:
            nlp_fill_ids = pd.read_csv(os.path.join(self.FEATURE_DIR,"fill_ecmo_ids.csv"))["SPM ID"]
            for i in nlp_fill_ids:
                df.loc[df.SPM_ID  == i, "dwh.Monitoring.ECMO_Therapie"] = 1


        def impute_column(col):
            method = col['method']
            col_name = col['name']
            if is_train:
                self.train_values[col_name] = df[col_name].dropna()
            ref = self.train_values[col_name]
            if method == 'median':
                df[col_name] = np.where(df[col_name].isna(), ref.median(), df[col_name])
            elif method == 'nv':
                missing_indices = df[col_name][df[col_name].isna()].index
                imputed_values = self.sample_normal_from_column(ref, len(missing_indices))
                df.loc[missing_indices, col_name] = imputed_values

        if impute_config:
            for col in impute_config:
                impute_column(col)

        default_dict = {
            'Nein': 0,
            'Ja': 1,
            'gut': 1,
            'normal': 1,
            'scharf': 1,
            'offen': 1,
            'unauffällig': 1,
            'keines': 1,
            'keine': 1,
            'atraumatisch': 0,
            'negativ': 0,
            'o.p.B.': 0,
        }
        for col in encode_config["binary"]:
            if "default" in col.keys():
                df[col['name']] = np.where(df[col['name']].isna(), default_dict[col["default"]],df[col['name']])

        def handle_high_missing_ratio(col):
            if sum(df[col].isnull()) / len(df) >= missing_threshold:
                df[col] = np.where(df[col].isnull(), 0, 1)
                self.high_missing_ratio_cols.append(col)

        if is_train:
            for col in df.columns:
                handle_high_missing_ratio(col)
        else:
            for col in self.high_missing_ratio_cols:
                df[col] = np.where(df[col].isnull(), 0, 1)

        # Check if there are more missing values to impute:
        n_missing = df.isnull().sum().sum()
        if n_missing == 0:
            return df
        
        # Prepare for iterative imputing
        no_impute_cols = ["SPM_ID", "Transplantationsstatus"]
        impute_df = df.drop(columns=no_impute_cols)
        if is_train:
            # Find all integer columns so they can be rounded back to integer after inverse scaling
            def check_int(col):
                return pd.api.types.is_numeric_dtype(col) and (col % 1 == 0).all()

            impute_df = impute_df.apply(lambda col: col.astype(int) if check_int(col) else col)
            self.int_cols = impute_df.select_dtypes("int").columns

            # scale for iterative imputing
            self.scaler = StandardScaler().fit(impute_df)

        scaled_df = self.scaler.transform(impute_df)

        # Iterative Imputing
        if is_train:
            br = Ridge()
            self.iterative_imp = IterativeImputer(
                estimator=br,
                verbose=2,
                max_iter=10,
                n_nearest_features=300,
                tol=1e-3,
                imputation_order="ascending",
                skip_complete=True,
                random_state=42,
            )
            self.iterative_imp.fit(scaled_df)
        imputed = self.iterative_imp.transform(scaled_df)
        imputed_df = pd.DataFrame(self.scaler.inverse_transform(imputed),
                                  columns=impute_df.columns)
        imputed_df.index = impute_df.index
        imputed_df[self.int_cols] = imputed_df[self.int_cols].round().astype(int) # Scaler turns int cols to floats
        df = pd.concat([df[no_impute_cols], imputed_df], axis=1)
        return df

    def drop_prefix(self, df):
        df.columns = [col.replace("dwh.Labor_Virologie.", "") \
                if "dwh.Labor_Virologie." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Labor_KCBU.", "") \
                        if "dwh.Labor_KCBU." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Transplantat.", "") \
                        if "dwh.Transplantat." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Labor_Mikrobiologie.", "") \
                        if "dwh.Labor_Mikrobiologie." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Labor_Pathologie.", "") \
                        if "dwh.Labor_Pathologie." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Abdomen.", "") \
                        if "dwh.Untersuchung_Abdomen." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Blutgase100.", "") \
                        if "dwh.Untersuchung_Blutgase100." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Bronchoskopie.", "") \
                        if "dwh.Untersuchung_Bronchoskopie." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_EKG.", "") \
                        if "dwh.Untersuchung_EKG." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Coronarangio.", "") \
                        if "dwh.Untersuchung_Coronarangio." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Echokardio.Herzfrequenz", "Echokardio.Herzfrequenz") \
                        if "dwh.Untersuchung_Echokardio.Herzfrequenz" in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Echokardio.", "") \
                        if "dwh.Untersuchung_Echokardio." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Koerper.", "") \
                        if "dwh.Untersuchung_Koerper." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_Thorax.", "") \
                        if "dwh.Untersuchung_Thorax." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Monitoring.Herzfrequenz", "Monitoring.Herzfrequenz") \
                        if "dwh.Monitoring.Herzfrequenz" in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Monitoring.", "") \
                        if "dwh.Monitoring." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Medikation.", "") \
                        if "dwh.Medikation." in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Untersuchung_", "") \
                        if "dwh.Untersuchung_" in col else col for col in list(df.columns)]

        df.columns = [col.replace("dwh.Spender.Spender_", "") \
                        if "dwh.Spender.Spender_" in col else col for col in list(df.columns)]
        return df

    def feature_engineering(self, df, is_train=True):
        """Remove duplicates and constant/duplicate features."""
        # Remove rows with duplicates or missing target
        df = df.drop_duplicates()
        df = df[~df[self.target].isna()]

        # Remove unnecessary prefixes
        # df = self.drop_prefix(df)

        # Basic feature selection
        x = df.drop(columns=self.target)
        y = df[self.target]
        if is_train:
            self.dfc = DropConstantFeatures(missing_values="include")
            self.dfc.fit(x)
        x = self.dfc.transform(x)
        if is_train:
            self.ddf = DropDuplicateFeatures()
            self.ddf.fit(x)
        x = self.ddf.transform(x)
        df = pd.concat([x, y], axis=1)
        return df
    
    def _preprocess(self, df, config, impute=True):
        """Apply preprocessing steps to the DataFrame."""
        df = self.filter_organ(df, config['organs'])

        # Scale, Create, Encode, Drop and Rename Features
        df = self._apply_optional_steps(df, config)

        # Combine Rows to have only one per Donor
        # True, if at least one kidney is transplanted
        target_df = df.groupby('SPM_ID', as_index=False).agg({
            'Transplantationsstatus': lambda x: any(x)
        })
        target_df['Transplantationsstatus'] = target_df['Transplantationsstatus'].astype(int)
        # Remove Transplantationsstatus, will be joined in target_df 
        df = df.drop(columns=['Transplantationsstatus'])
        # Only keep first row per SPM_ID (Usually same values, 
        # otherwise it was checked that this makes sense)
        df = df.drop_duplicates(subset='SPM_ID', keep='first')
        # Join new Transplantationsstatus to df
        df = pd.merge(target_df, df, on='SPM_ID', how='left')

        
        df_train, df_test = self.split_by_ids(df)

        # Impute data
        if impute:
            # Impute training data
            df_train = self.impute(df_train, impute_config=config["impute"],
                                   encode_config=config["encode"], is_train=True)
            # Impute test data using the same parameters
            df_test = self.impute(df_test, impute_config=config["impute"],
                                  encode_config=config["encode"], is_train=False)
            
        # Remove rows, prefixes and basic feature selection
        df_train = self.feature_engineering(df_train, is_train=True)
        df_test = self.feature_engineering(df_test, is_train=False)

        # Combine, in case test split is not wanted
        df = pd.concat([df_train, df_test], axis=0)
        df = self.move_column_to_last(df, self.target)
        return df

    def perform(self, config_path, test_split=True, dev=False, impute=True):
        """Execute complete preprocessing pipeline based on configuration."""
        # Load data and config
        config = self._load_config(config_path)
        df = pd.read_pickle(os.path.join(self.FEATURE_DIR, "dev", "merged.pkl"))

        # Filter for columns in config
        df = self.filter_columns(df, config)

        # Execute preprocessing
        final_df = self._preprocess(df, config, impute=impute)
        
        # Save results
        print('Saving Data...')
        dataset_path = self._prepare_saving(config['new_dataset_name'], dev)
        clear_csv_files(dataset_path)
        self._save_data(final_df, dataset_path, test_split)

if __name__ == "__main__":
    processor = DataPreprocessor(target='Transplantationsstatus')
    processor.perform('src/data/exploration_config.yml',
                      test_split=True,
                      impute=True,
                      dev=False)
