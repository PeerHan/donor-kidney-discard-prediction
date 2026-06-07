import os
import warnings
import re
from dotenv import load_dotenv
import yaml

from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

from utils import clear_csv_files

# Clean up console output
warnings.filterwarnings("ignore", category=DeprecationWarning)
#warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
# Downcasting object dtype arrays on .fillna, .ffill, .bfill is deprecated and will change in a future version.
pd.set_option('future.no_silent_downcasting', True)


# Merger Object
class DataMerger:
    def __init__(self):
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

    def process_target(self, df, method):
        """Process target variable according to specified method."""
        if method == '3_targets':
            df['Transplantationsstatus'] = df['dwh.Transplantat.Status_Transplantat'].apply(
                lambda x: (
                    "Transplantiert"
                    if x == "transplantiert"
                    else (
                        "Nicht transplantiert"
                        if x
                        in [
                            "Abbruch vor ET-Meldung",
                            "Ausschluss von ET bestätigt",
                            "Nicht transplantiert",
                            "nach ET-Meldung ausgeschlossen",
                        ]
                        else "Nicht vermittelbar" if x == "Nicht vermittelbar" else None
                    )
                )
            )
            df = df.drop(columns="dwh.Transplantat.Status_Transplantat")
        elif method == "transplantiert_oder_nicht":
            df["Transplantationsstatus"] = df[
                "dwh.Transplantat.Status_Transplantat"
            ].apply(lambda x: x == "transplantiert").astype(int)
            df = df.drop(columns="dwh.Transplantat.Status_Transplantat")
        return df

    def combine_columns(self, df, combine_configs):
        combined_df = df[['SPM_ID']].drop_duplicates().reset_index(drop=True)

        for config in combine_configs:
            columns = config['columns']
            method = config['method']
            new_name = config['new_name']

            if method == 'lowest':
                temp_df = df.groupby('SPM_ID')[columns].min().reset_index()
                temp_df[new_name] = temp_df[columns].min(axis=1, skipna=True)
            elif method == 'max':
                temp_df = df.groupby('SPM_ID')[columns].max().reset_index()
                temp_df[new_name] = temp_df[columns].max(axis=1, skipna=True)
            elif method == 'first':
                temp_df = df.groupby('SPM_ID')[columns].first().reset_index()
                temp_df[new_name] = temp_df[columns].apply(
                    lambda row: next((x for x in row if pd.notna(x)), np.nan),
                    axis=1
                )

            temp_df = temp_df[['SPM_ID', new_name]]
            combined_df = pd.merge(combined_df, temp_df, on='SPM_ID', how='left')

        # Drop the original columns after processing
        df.drop(columns=[col for config in combine_configs for col in config['columns']], inplace=True)

        return combined_df

    def group_by_columns(self, sheet_config, sheet_df):
        group_by_columns = sheet_config['group_by_columns']
        aggregated_df = sheet_df["SPM_ID"].drop_duplicates()
        first_valid_df = sheet_df["SPM_ID"].drop_duplicates()
        latest_date_df = sheet_df["SPM_ID"].drop_duplicates()
        recent_valid_df = sheet_df["SPM_ID"].drop_duplicates()

        if 'aggregate_columns' in sheet_config.keys():
            aggregate_columns = sheet_config['aggregate_columns']
            aggregated_df = (sheet_df.groupby(group_by_columns)[aggregate_columns]
                            .apply(lambda x: (x == sheet_config['aggregate_value']).sum())
                            .reset_index())
            
            aggregated_df = aggregated_df[group_by_columns + aggregate_columns]

        if "first_valid_columns" in sheet_config.keys():
            first_valid_columns = sheet_config["first_valid_columns"]

            first_valid_df = sheet_df.groupby(group_by_columns).agg({
                col: lambda x: x.dropna().iloc[0] if not x.dropna().empty else np.nan
                for col in first_valid_columns
            }).reset_index()

            first_valid_df = first_valid_df[group_by_columns + first_valid_columns]

        if 'latest_date_columns' in sheet_config.keys():
            latest_date_columns = sheet_config['latest_date_columns']
            date_column = sheet_config['date_column'][0]

            # Relevante Spalten auswählen
            columns_to_keep = group_by_columns + [date_column] + latest_date_columns
            latest_date_df = sheet_df[columns_to_keep]

            # Entfernen der Zeilen, bei denen kein Untersuchungsdatum angegeben ist
            sheet_df1 = latest_date_df.dropna(subset=date_column)

            # Filtere Zeilen, bei denen der Wert in der Spalte NaT ist
            sheet_df2 = latest_date_df[latest_date_df[date_column].isna()]

            # Finden der Indzies mit dem aktuellsten Datum pro Gruppierung
            sheet_df1 = sheet_df1.loc[sheet_df1.groupby(group_by_columns)[date_column].idxmax()]

            # Kombiniere die beiden DataFrames untereinander
            latest_date_df = pd.concat([sheet_df1, sheet_df2], ignore_index=True)

            # Datum Spalte entfernen
            latest_date_df = latest_date_df[group_by_columns + latest_date_columns]

        if 'recent_valid_columns' in sheet_config.keys():
            recent_valid_columns = sheet_config['recent_valid_columns']
            date_column = sheet_config['date_column'][0]
            number_days = sheet_config['number_days'][0]

            # Entfernen der Zeilen, bei denen kein Untersuchungsdatum angegeben ist
            sheet_df1 = sheet_df[group_by_columns + [date_column] + recent_valid_columns].dropna(subset=date_column)

            # Filtere Zeilen, bei denen der Wert in der Spalte NaT ist
            sheet_df2 = sheet_df[sheet_df[date_column].isna()]

            # Entfernen von Zeilen, deren Datum mehr als n Tage älter als das aktuellste ist
            latest_dates = sheet_df1.groupby(group_by_columns)[date_column].transform('max')
            sheet_df1 = sheet_df1[sheet_df1[date_column] >= latest_dates - pd.Timedelta(days=number_days)]

            # Funktion, um fehlende Werte basierend auf dem nächsten gültigen Wert zu füllen
            def find_nearest_valid_value(group):
                for col in recent_valid_columns:
                    if group[col].isna().any():
                        # Sortiere nach Datum absteigend
                        sorted_group = group.sort_values(by=date_column, ascending=False)
                        for _, row in sorted_group.iterrows():
                            if not pd.isna(row[col]):
                                group[col] = group[col].fillna(row[col])
                                break
                return group

            # Gruppieren und Anwenden der Fülllogik
            sheet_df1 = (
                sheet_df1.groupby(group_by_columns, group_keys=False).apply(find_nearest_valid_value)
            )

            # Nur die aktuellste Zeile pro Gruppierung behalten
            sheet_df1 = sheet_df1.loc[sheet_df1.groupby(group_by_columns)[date_column].idxmax()]

            # Kombiniere die beiden DataFrames untereinander
            recent_valid_df = pd.concat([sheet_df1, sheet_df2], ignore_index=True)

            # Datum Spalte entfernen
            recent_valid_df = recent_valid_df[group_by_columns + recent_valid_columns]

        sheet_df = pd.merge(aggregated_df, first_valid_df, on=group_by_columns, how="left")
        sheet_df = pd.merge(sheet_df, latest_date_df, on=group_by_columns, how="left")
        sheet_df = pd.merge(sheet_df, recent_valid_df, on=group_by_columns, how="left")
        return sheet_df

    def _clean_expression(self, text):
        """Clean text"""
        # If the text contains "sonstige blutprodukte", "fresh frozen plasma", or "human albumin", keep it unchanged
        keywords = ["sonstige blutprodukte", "fresh frozen plasma", "human albumin"]
        if any(keyword in text.lower() for keyword in keywords):
            return text.strip()

        # Remove all content after specific characters
        #  --> keeps only the first word, as it is usually the medication name
        cleaned_text = re.sub(r'[ ,®/].*', '', text)

        return cleaned_text.strip()

    def _specific_replacements(self, text):
        """Function for specific replacements"""
        replacements = {
            "huminsulin": "insulin",
            "methylprednisolut": "methylprednisolone",
            "urbason": "methylprednisolone",
            "heparin-natrium": "heparin",
            "actrapid": "insulin",
            "insuman": "insulin",
            "berlinsulin": "insulin",
            "arterenol": "noradrenaline",
        }
        # Check if a replacement is necessary; otherwise, return the original text
        for key, value in replacements.items():
            if re.search(fr'\b{key}\b', text, flags=re.IGNORECASE):
                return re.sub(fr'\b{key}\b', value, text, flags=re.IGNORECASE)

        return text  # Return original text if no replacements were applied

    def _process_diabetes_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in diabetes sheet."""
        dummies_list = []

        # Create dummies for ICD codes
        if 'Diabetes_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Diabetes_ICD_10_Code'],
                                    prefix='Diabetes_ICD_10_Code',
                                    dummy_na=False).astype(int)
            dummies_list.append(icd_dummies)
        # Create dummies for Graduierung if the column exists
        if 'Diabetes_Behandlung' in sheet_df.columns:
            grad_dummies = pd.get_dummies(sheet_df['Diabetes_Behandlung'],
                                          prefix='Diabetes_Behandlung',
                                          dummy_na=False).astype(int)
            dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Check and include 'Alkohol_Diagnose_Zeitpunkt_von' and 'Alkohol_Diagnose_Zeitpunkt_bis' if they exist
            if 'Diabetes_Diagnose_Zeitpunkt_von' in sheet_df.columns:
                sheet_df['Diabetes_Diagnose_Zeitpunkt_von'] = pd.to_datetime(
                    sheet_df['Diabetes_Diagnose_Zeitpunkt_von']
                    )
            if 'Diabetes_Diagnose_Zeitpunkt_bis' in sheet_df.columns:
                sheet_df['Diabetes_Diagnose_Zeitpunkt_bis'] = pd.to_datetime(
                    sheet_df['Diabetes_Diagnose_Zeitpunkt_bis']
                    )

            # Group by SPM_ID and aggregate using maximum value for dummies
            # and min/max for the diagnosis dates
            result_df = (
                dummies_df
                .assign(
                    SPM_ID=sheet_df.get('SPM_ID'),
                    Diabetes_Diagnose_Zeitpunkt_von=sheet_df.get('Diabetes_Diagnose_Zeitpunkt_von'),
                    Diabetes_Diagnose_Zeitpunkt_bis=sheet_df.get('Diabetes_Diagnose_Zeitpunkt_bis'),
                    Diabetes_Diagnose=sheet_df.get('Diabetes_Diagnose')
                )
                .groupby('SPM_ID')
                .agg({**{col: 'max' for col in dummies_df.columns},
                    'Diabetes_Diagnose_Zeitpunkt_von': 'min',
                    'Diabetes_Diagnose_Zeitpunkt_bis': 'max',
                    'Diabetes_Diagnose': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None})
                .reset_index()
            )

        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df


        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if col.startswith('Diabetes_ICD_10_Code') or col.startswith('Diabetes_Behandlung')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_sepsis_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in sepsis sheet."""
        dummies_list = []

        # Create dummies for ICD codes
        if 'Sepsis_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Sepsis_ICD_10_Code'],
                                    prefix='Sepsis_ICD_10_Code',
                                    dummy_na=False)
            dummies_list.append(icd_dummies)
        # Create dummies for Graduierung if the column exists
        if 'Sepsis_Bewertung_Behandlungserfolg' in sheet_df.columns:
            grad_dummies = pd.get_dummies(sheet_df['Sepsis_Bewertung_Behandlungserfolg'],
                                          prefix='Sepsis_Bewertung_Behandlungserfolg',
                                          dummy_na=False)
            dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Group by SPM_ID and aggregate using maximum value
            # This ensures if any drug type or graduierung exists, it will be marked as 1
            result_df = dummies_df.groupby(sheet_df['SPM_ID']).max().reset_index()
            # Check and include 'Alkohol_Diagnose_Zeitpunkt_von' and 'Alkohol_Diagnose_Zeitpunkt_bis' if they exist
            if 'Sepsis_Diagnose_Zeitpunkt_von' in sheet_df.columns:
                sheet_df['Sepsis_Diagnose_Zeitpunkt_von'] = pd.to_datetime(sheet_df['Sepsis_Diagnose_Zeitpunkt_von'])
            if 'Sepsis_Diagnose_Zeitpunkt_bis' in sheet_df.columns:
                sheet_df['Sepsis_Diagnose_Zeitpunkt_bis'] = pd.to_datetime(sheet_df['Sepsis_Diagnose_Zeitpunkt_bis'])

            # Group by SPM_ID and aggregate using maximum value for dummies
            # and min/max for the diagnosis dates
            result_df = (
                dummies_df
                .assign(
                    Sepsis_Diagnose_Zeitpunkt_von=sheet_df.get('Sepsis_Diagnose_Zeitpunkt_von'),
                    Sepsis_Diagnose_Zeitpunkt_bis=sheet_df.get('Sepsis_Diagnose_Zeitpunkt_bis')
                )
                .groupby(sheet_df['SPM_ID'])
                .agg({**{col: 'max' for col in dummies_df.columns},
                    'Sepsis_Diagnose_Zeitpunkt_von': 'min',
                    'Sepsis_Diagnose_Zeitpunkt_bis': 'max'})
                .reset_index()
            )
        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df

        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if col.startswith('Sepsis_ICD_10_Code')
                         or col.startswith('Sepsis_Bewertung_Behandlungserfolg')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_rauchen_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in rauchen sheet."""
        dummies_list = []

        # Create dummies for ICD codes
        if 'Rauchen_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Rauchen_ICD_10_Code'],
                                         prefix='Rauchen_ICD_10_Code',
                                         dummy_na=False)
            dummies_list.append(icd_dummies)

        # Create dummies for Graduierung if the column exists
        if 'Graduierung_Raucherklassifikation' in sheet_df.columns:
            grad_dummies = pd.get_dummies(sheet_df['Graduierung_Raucherklassifikation'],
                                          prefix='Graduierung_Raucherklassifikation',
                                          dummy_na=False).astype(int)
            dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Group by SPM_ID and aggregate using maximum value
            # This ensures if any drug type or graduierung exists, it will be marked as 1
            result_df = dummies_df.groupby(sheet_df['SPM_ID']).max().reset_index()
        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df

        # Group by SPM_ID and aggregate using maximum value for dummies
        # and min/max for the diagnosis dates
        # result_df = (
        #     icd_dummies
        #     .assign(
        #         Alkohol_Diagnose_Zeitpunkt_von=sheet_df.get('Alkohol_Diagnose_Zeitpunkt_von'),
        #         Alkohol_Diagnose_Zeitpunkt_bis=sheet_df.get('Alkohol_Diagnose_Zeitpunkt_bis')
        #     )
        #     .groupby(sheet_df['SPM_ID'])
        #     .agg({**{col: 'max' for col in icd_dummies.columns},
        #           'Alkohol_Diagnose_Zeitpunkt_von': 'min',
        #           'Alkohol_Diagnose_Zeitpunkt_bis': 'max'})
        #     .reset_index()
        # )

        # Handle numerical columns by taking the first non-NaN value
        numerical_columns = ['Pack_years', 'Anzahl_Zigarettenschachteln_pro_Tag', 'Dauer_Zigaretten_Konsum']
        for col in numerical_columns:
            if col in sheet_df.columns:
                result_df[col] = sheet_df.groupby('SPM_ID')[col].apply(
                    lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
                    )

        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if col.startswith('Rauchen_ICD_10_Code')
                         or col.startswith('Graduierung_Raucherklassifikation')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_malignom_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in malignom sheet."""
        dummies_list = []

        # Create dummies for ICD codes
        if 'Malignom_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Malignom_ICD_10_Code'],
                                    prefix='Malignom_ICD_10_Code',
                                    dummy_na=False).astype(int)
            dummies_list.append(icd_dummies)
        # Create dummies for Graduierung if the column exists
        categorical_columns = ['Grading',
                               'Behandlung_abgeschlossen',
                               'Operation_erfolgt',
                               'Chemotherapie_erfolgt',
                               'Strahlentherapie',
                               'Rezidiv_bekannt',
                               'Metastasen_bekannt',
                               'Nachsorge_erfolgt',
                               'Remission']
        for col in categorical_columns:
            if col in sheet_df.columns:
                grad_dummies = pd.get_dummies(sheet_df[col],
                                              prefix=col,
                                              dummy_na=False)
                dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Group by SPM_ID and aggregate using maximum value
            # This ensures if any drug type or graduierung exists, it will be marked as 1
            result_df = dummies_df.groupby(sheet_df['SPM_ID']).max().reset_index()
        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df
        # Check for 'Gesamtstrahlendosis' and take the first non-NaN value for each SPM_ID
        if 'Gesamtstrahlendosis' in sheet_df.columns:
            gesamtstrahlendosis_series = sheet_df.groupby('SPM_ID')['Gesamtstrahlendosis'].apply(
                lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
                )
            result_df = result_df.merge(gesamtstrahlendosis_series, on='SPM_ID', how='left')

        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if any(col.startswith(prefix) for prefix in categorical_columns)]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_hypertonie_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in hypertonie sheet."""
        dummies_list = []

        # Create dummies for ICD codes
        if 'Hypertonie_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Hypertonie_ICD_10_Code'],
                                    prefix='Hypertonie_ICD_10_Code',
                                    dummy_na=False)
            dummies_list.append(icd_dummies)
        # Create dummies for Graduierung if the column exists
        if 'Hypertonie_behandelt' in sheet_df.columns:
            grad_dummies = pd.get_dummies(sheet_df['Hypertonie_behandelt'],
                                          prefix='Hypertonie_behandelt',
                                          dummy_na=False)
            dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Group by SPM_ID and aggregate using maximum value
            # This ensures if any drug type or graduierung exists, it will be marked as 1
            result_df = dummies_df.groupby(sheet_df['SPM_ID']).max().reset_index()
        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df
        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if col.startswith('Hypertonie_behandelt') or col.startswith('Hypertonie_ICD_10_Code')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_ohne_diagnosegruppe_sheet(self, sheet_df, sheet_config, threshold=0.01):
        """Handle multiple entries per SPM_ID in ohne diagnosegruppe sheet."""
        icd_dummies = None

        # Create dummies for ICD codes
        if 'Ohne_Diagnosegruppe_ICD_10_Code' in sheet_df.columns:
            # Calculate value counts and percentages
            value_counts = sheet_df['Ohne_Diagnosegruppe_ICD_10_Code'].value_counts()
            total_rows = len(sheet_df)
            percentages = value_counts / total_rows

            # Apply threshold to filter out rare codes
            min_percentage = sheet_config.get('min_percentage', threshold)
            code_mapping = {
                code: code if percentage >= min_percentage else np.nan
                for code, percentage in percentages.items()
            }

            # Map the codes based on the threshold
            sheet_df['Ohne_Diagnosegruppe_ICD_10_Code'] = sheet_df['Ohne_Diagnosegruppe_ICD_10_Code'].map(code_mapping)
            icd_dummies = pd.get_dummies(sheet_df['Ohne_Diagnosegruppe_ICD_10_Code'],
                                    prefix='Ohne_Diagnosegruppe_ICD_10_Code',
                                    dummy_na=False)

        # Combine all dummy columns if any exist
        if icd_dummies is not None:
            # Group by SPM_ID and aggregate using maximum value
            # This ensures if any drug type or graduierung exists, it will be marked as 1
            result_df = icd_dummies.groupby(sheet_df['SPM_ID']).max().reset_index()
        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df
        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns if col.startswith('Ohne_Diagnosegruppe_ICD_10_Code')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_drogen_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in drug diagnosis sheet."""
        dummies_list = []

        # Create dummies for ICD codes
        if 'Drogen_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Drogen_ICD_10_Code'],
                                    prefix='Drogen_ICD_10_Code',
                                    dummy_na=False)
            dummies_list.append(icd_dummies)

        # Create dummies for Graduierung if the column exists
        if 'Graduierung_Drogenkonsum' in sheet_df.columns:
            grad_dummies = pd.get_dummies(sheet_df['Graduierung_Drogenkonsum'],
                                          prefix='Graduierung_Drogenkonsum',
                                          dummy_na=False).astype(int)
            dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Group by SPM_ID and aggregate using maximum value
            # This ensures if any drug type or graduierung exists, it will be marked as 1
            result_df = dummies_df.assign(
                SPM_ID=sheet_df.get('SPM_ID'),
                Drogen_Diagnose=sheet_df.get('Drogen_Diagnose')
            ).groupby('SPM_ID').agg({
                **{col: 'max' for col in dummies_df.columns},
                'Drogen_Diagnose': lambda x: x.dropna().iloc[0] if not x.dropna().empty else None
            }).reset_index()
        else:
            # If no dummies were created, return an empty DataFrame with SPM_ID
            result_df = sheet_df

        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if col.startswith('Drogen_ICD_10_Code') or col.startswith('Graduierung_Drogenkonsum')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_alkohol_sheet(self, sheet_df):
        """Handle multiple entries per SPM_ID in alcohol diagnosis sheet."""
        dummies_list = []

        # Check and create dummies for ICD codes if the column exists
        if 'Alkohol_ICD_10_Code' in sheet_df.columns:
            icd_dummies = pd.get_dummies(sheet_df['Alkohol_ICD_10_Code'],
                                         prefix='Alkohol_ICD',
                                         dummy_na=False)
            dummies_list.append(icd_dummies)

        # Check and create dummies for Graduierung if the column exists
        if 'Graduierung_Alkoholkonsum' in sheet_df.columns:
            grad_dummies = pd.get_dummies(sheet_df['Graduierung_Alkoholkonsum'],
                                          prefix='Alkohol_Graduierung',
                                          dummy_na=False)
            dummies_list.append(grad_dummies)

        # Combine all dummy columns if any exist
        if dummies_list:
            dummies_df = pd.concat(dummies_list, axis=1)
            # Group by SPM_ID and aggregate using maximum value for dummies
            # and min/max for the diagnosis dates

            # Ensure 'Alkohol_Diagnose_Zeitpunkt_von' and 'Alkohol_Diagnose_Zeitpunkt_bis' are datetime
            if 'Alkohol_Diagnose_Zeitpunkt_von' in sheet_df.columns:
                sheet_df['Alkohol_Diagnose_Zeitpunkt_von'] = pd.to_datetime(
                    sheet_df['Alkohol_Diagnose_Zeitpunkt_von'], errors='coerce'
                    )
            if 'Alkohol_Diagnose_Zeitpunkt_bis' in sheet_df.columns:
                sheet_df['Alkohol_Diagnose_Zeitpunkt_bis'] = pd.to_datetime(
                    sheet_df['Alkohol_Diagnose_Zeitpunkt_bis'], errors='coerce'
                    )

            result_df = (
                dummies_df
                .assign(
                    Alkohol_Diagnose_Zeitpunkt_von=sheet_df.get('Alkohol_Diagnose_Zeitpunkt_von'),
                    Alkohol_Diagnose_Zeitpunkt_bis=sheet_df.get('Alkohol_Diagnose_Zeitpunkt_bis')
                )
                .groupby(sheet_df['SPM_ID'])
                .agg({**{col: 'max' for col in dummies_df.columns},
                    'Alkohol_Diagnose_Zeitpunkt_von': 'min',
                    'Alkohol_Diagnose_Zeitpunkt_bis': 'max'})
                .reset_index()
            )
        else:
            dummies_df = pd.DataFrame(index=sheet_df.index)

        # Convert boolean dummies to integers
        dummy_columns = [col for col in result_df.columns
                         if col.startswith('Alkohol_ICD') or col.startswith('Alkohol_Graduierung')]
        result_df[dummy_columns] = result_df[dummy_columns].astype(int)

        return result_df

    def _process_allokation_sheet(self, sheet_df):
        sheet_df = sheet_df.loc[sheet_df['dwh.Transplantat.dwh.Transplantat_Allokation.Transplantat']
                                .str.startswith('Niere', na=False)].reset_index(drop=True)
        sheet_df = sheet_df.drop(columns=['dwh.Transplantat.dwh.Transplantat_Allokation.Transplantat'])
        return sheet_df

    def _summarize_and_regress(self, group, column):
        """Function to compute time-series features and regression for a single SPM_ID"""
        X = group['Hours Since First Measurement'].values.reshape(-1, 1)  # Explanatory variable
        y = group[column].values  # Dependent variable

        # Fit linear regression
        model = LinearRegression()
        model.fit(X, y)

        # Extract regression coefficients
        intercept = model.intercept_  # Intercept (b_0)
        slope = model.coef_[0]       # Linear term (b_1)

        # Calculate time-series features
        first_value = y[0] if len(y) > 0 else None
        last_value = y[-1] if len(y) > 0 else None
        value_count = len(y)
        std_deviation = np.std(y, ddof=0) if len(y) > 1 else 0
        total_hours = X[-1][0] if len(X) > 0 else None
        min_value = y.min() if len(y) > 0 else None
        max_value = y.max() if len(y) > 0 else None

        # Return the results as a dictionary
        return {
            f'{column}_intercept': intercept,
            f'{column}_slope': slope,
            f'{column}_first_value': first_value,
            f'{column}_last_value': last_value,
            f'{column}_value_count': value_count,
            f'{column}_std_deviation': std_deviation,
            f'{column}_total_hours': total_hours,
            f'{column}_min_value': min_value,
            f'{column}_max_value': max_value
        }

    def _summarize(self, group, column):
        """Function to compute time-series features for a single SPM_ID"""
        X = group['Hours Since First Measurement'].values.reshape(-1, 1)  # Explanatory variable
        y = group[column].values  # Dependent variable

        # Calculate time-series features
        first_value = y[0] if len(y) > 0 else None
        last_value = y[-1] if len(y) > 0 else None
        value_count = len(y)
        std_deviation = np.std(y, ddof=0) if len(y) > 1 else 0
        total_hours = X[-1][0] if len(X) > 0 else None
        min_value = y.min() if len(y) > 0 else None
        max_value = y.max() if len(y) > 0 else None

        # Return the results as a dictionary
        return {
            f'{column}_first_value': first_value,
            f'{column}_last_value': last_value,
            f'{column}_value_count': value_count,
            f'{column}_std_deviation': std_deviation,
            f'{column}_total_hours': total_hours,
            f'{column}_min_value': min_value,
            f'{column}_max_value': max_value
        }

    def _summarize_qualitative(self, group, column):
        """Function to compute time-series features for a single SPM_ID"""
        X = group['Hours Since First Measurement'].values.reshape(-1, 1)  # Explanatory variable
        y = group[column].values  # Dependent variable

        # Calculate time-series features
        first_value = y[0] if len(y) > 0 else None
        last_value = y[-1] if len(y) > 0 else None
        value_count = len(y)
        total_hours = X[-1][0] if len(X) > 0 else None

        # Return the results as a dictionary
        return {
            f'{column}_first_value': first_value,
            f'{column}_last_value': last_value,
            f'{column}_value_count': value_count,
            f'{column}_total_hours': total_hours
        }

    def process_excel_file(self, file_path, sheet_configs):
        """Process each sheet in the Excel file based on configuration."""
        merged_df = None
        sheet_processing_map = {
            'dwh Transplantat': self._process_transplantat_sheet,
            'dwh Diagnose Drogen': self._process_drogen_sheet,
            'dwh Diagnose Alkohol': self._process_alkohol_sheet,
            'dwh Transplantat Allokation': self._process_allokation_sheet,
            'dwh Diagnose ohne Gruppe': self._process_ohne_diagnosegruppe_sheet,
            'dwh Diagnose Hypertonie': self._process_hypertonie_sheet,
            'dwh Diagnose Diabetes': self._process_diabetes_sheet,
            'dwh Diagnose Rauchen': self._process_rauchen_sheet,
            'dwh Diagnose Sepsis': self._process_sepsis_sheet,
            'dwh Diagnose Malignom': self._process_malignom_sheet,
        }

        for sheet_config in tqdm(sheet_configs):
            sheet_name = sheet_config['name']

            # Read the sheet with a single conditional block
            def read_sheet(sheet_config, sheet_name):
                if 'columns_to_keep' in sheet_config:
                    cols = ['SPM_ID'] + sheet_config['columns_to_keep']
                    return pd.read_excel(file_path, sheet_name=sheet_name, usecols=cols)
                if 'columns_to_drop' in sheet_config:
                    def make_usecols(cols_to_drop):
                        return lambda x: x not in cols_to_drop
                    usecols = make_usecols(sheet_config['columns_to_drop'])
                    return pd.read_excel(file_path, sheet_name=sheet_name, usecols=usecols)
                return pd.read_excel(file_path, sheet_name=sheet_name)

            # Define transformation functions that encapsulate related logic
            def transform_post_admission(df, sheet_config):
                post_admission_date = sheet_config['post_admission_date'][0]
                excel_file_diag = pd.ExcelFile(os.getenv('Diag_20_24'))
                df_admission = pd.read_excel(excel_file_diag, 'dwh Diagonose Aufnahme')
                df = df.merge(df_admission[['SPM_ID', 'Aufnahme_Diagnose_Zeitpunkt_von']],
                              on='SPM_ID', how='left')
                df1 = df[(df['Aufnahme_Diagnose_Zeitpunkt_von'].isna()) |
                         (df[post_admission_date].isna()) |
                         (df[post_admission_date] >= df['Aufnahme_Diagnose_Zeitpunkt_von'])]
                df2 = df[df[post_admission_date] < df['Aufnahme_Diagnose_Zeitpunkt_von']]
                df2 = df2[~df2['SPM_ID'].isin(df1['SPM_ID'])]
                cols = ['SPM_ID', post_admission_date, 'Aufnahme_Diagnose_Zeitpunkt_von']
                df2.loc[:, df2.columns.difference(cols)] = np.nan
                return pd.concat([df1, df2], ignore_index=True).drop(columns=['Aufnahme_Diagnose_Zeitpunkt_von'])

            def transform_pre_op(df, sheet_config):
                pre_op_date = sheet_config['pre_op_date'][0]
                excel_file_op = pd.ExcelFile(os.getenv('Op_20_24'))
                df_op = pd.read_excel(excel_file_op, 'dwh OP Explantation')
                df = df.merge(df_op[['SPM_ID', 'dwh.OPExplantation.Cross_Clamp_Zeitpunkt']],
                              on='SPM_ID', how='left')
                df1 = df[(df['dwh.OPExplantation.Cross_Clamp_Zeitpunkt'].isna()) |
                         (df[pre_op_date].isna()) |
                         (df[pre_op_date] < df['dwh.OPExplantation.Cross_Clamp_Zeitpunkt'])]
                df2 = df[df[pre_op_date] >= df['dwh.OPExplantation.Cross_Clamp_Zeitpunkt']]
                df2 = df2[~df2['SPM_ID'].isin(df1['SPM_ID'])]
                cols = ['SPM_ID', pre_op_date, 'dwh.OPExplantation.Cross_Clamp_Zeitpunkt']
                df2.loc[:, df2.columns.difference(cols)] = np.nan
                return pd.concat([df1,
                                  df2],
                                  ignore_index=True).drop(columns=['dwh.OPExplantation.Cross_Clamp_Zeitpunkt'])

            def transform_sort(df, sheet_config):
                return df.sort_values(by=sheet_config['sort_by_columns'], ascending=[True, False])

            def transform_combine(df, sheet_config):
                return self.combine_columns(df, sheet_config['combine'])

            def transform_clean(df, sheet_config):
                cols = sheet_config['columns_to_clean']
                repl = sheet_config['replace_values']
                df[cols] = df[cols].replace(repl, np.nan)
                return df

            def transform_filter_column(df, sheet_config):
                col = sheet_config['filter_column'][0]
                keep = sheet_config['keep_values']
                df[col] = df[col].where(df[col].isin(keep), np.nan)
                return df

            def transform_col_filter(df, sheet_config):
                col = sheet_config['column_to_filter'][0]
                val = sheet_config['value_to_drop'][0]
                return df[df[col] != val].drop(columns=[col])

            def transform_allowed_values(df, sheet_config):
                col = sheet_config['column_with_allowed_values'][0]
                allowed = sheet_config['allowed_values']
                cond = ~df[col].isin(allowed)
                upd_cols = df.columns.difference(['SPM_ID'])
                df.loc[cond, upd_cols] = np.nan
                return df

            def transform_multiply(df, sheet_config):
                cols = sheet_config['columns_to_multiply']
                base = sheet_config['based_on_column'][0]
                grouped = {}
                for value in df[base].dropna().unique():
                    tmp = df[df[base] == value].copy()
                    tmp.rename(
                        columns={c: f"{c}_{value.replace(' ', '_')}" for c in cols},
                        inplace=True
                    )
                    grouped[value] = tmp
                grouped['NaN'] = df[df[base].isna()].drop(columns=cols)
                return pd.concat(grouped.values(), axis=0, ignore_index=True, sort=True)

            def transform_cols_combine(df, sheet_config):
                cols = sheet_config['columns_to_combine']
                new_name = sheet_config['new_column_name'][0]
                df[new_name] = df[cols].bfill(axis=1).iloc[:, 0]
                return df.drop(columns=cols)

            def transform_med_map(df, sheet_config):
                col = sheet_config['med_column_to_map'][0]
                nan_df = df[df[col].isna()]
                df = df.dropna()
                df[col] = df[col].str.lower()\
                                .apply(self._clean_expression)\
                                .apply(self._specific_replacements)
                return df.merge(nan_df[['SPM_ID']], on='SPM_ID', how='outer')

            def transform_encode(df, sheet_config):
                col = sheet_config['column_to_encode']
                prefix = sheet_config['encoding_prefix'][0]
                one_hot = pd.get_dummies(df[col], prefix=prefix, prefix_sep='').astype(int)
                df = pd.concat([df[['SPM_ID']], one_hot], axis=1)
                return df.groupby('SPM_ID').max().reset_index()

            def transform_top_features(df, sheet_config):
                num = sheet_config['num_top_features'][0]
                sums = df.drop(columns=['SPM_ID']).sum()
                top_cols = sums.nlargest(num).index
                return df[['SPM_ID'] + list(top_cols)]

            def transform_group_by(df, sheet_config):
                return self.group_by_columns(sheet_config, df)

            def transform_timeseries(df, sheet_config):
                ts_reg = sheet_config['ts_to_summarize_and_regress']
                ts = sheet_config['ts_to_summarize']
                q_ts = sheet_config['qualitative_ts_to_summarize']
                time_col = sheet_config['time_column'][0]
                results = {}
                for col in ts_reg + ts + q_ts:
                    if col == 'dwh.Labor_KCBU.KC_C_reaktives_Protein_konkateniert':
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                    fdf = df[~df[col].isna()].copy()
                    if col in q_ts:
                        fdf[col] = fdf[col].replace({'Neg': 0, '-': 0, '(+)': 0,
                                                      '+++': 1, '+': 1, '++': 1, 'Pos': 1})
                    fdf['First Measurement'] = fdf.groupby('SPM_ID')[time_col].transform('min')
                    fdf['Hours Since First Measurement'] = (
                        (fdf[time_col] - fdf['First Measurement']).dt.total_seconds() / 3600
                    )
                    fdf = fdf.drop(columns=['First Measurement'])\
                           .sort_values(by=['SPM_ID', 'Hours Since First Measurement'], ascending=[True, True])
                    if col in ts_reg:
                        summary = fdf.groupby('SPM_ID')\
                                   .apply(lambda group, col=col: self._summarize_and_regress(group, col))
                    elif col in ts:
                        summary = fdf.groupby('SPM_ID')\
                                   .apply(lambda group, col=col: self._summarize(group, col))
                    else:
                        summary = fdf.groupby('SPM_ID')\
                                   .apply(lambda group, col=col: self._summarize_qualitative(group, col))
                    results[col] = pd.DataFrame(summary.tolist(), index=summary.index)
                final = pd.concat(results.values(), axis=1).reset_index()
                spm_df = df[~df['SPM_ID'].isin(final['SPM_ID'])][['SPM_ID']]\
                         .drop_duplicates(subset=['SPM_ID'], keep='first')
                return final.merge(spm_df, on='SPM_ID', how='outer')

            # Group the transformation functions into a single iterative loop
            transformations = [
                ('post_admission_date', transform_post_admission),
                ('pre_op_date', transform_pre_op),
                ('sort_by_columns', transform_sort),
                ('combine', transform_combine),
                ('columns_to_clean', transform_clean),
                ('filter_column', transform_filter_column),
                ('column_to_filter', transform_col_filter),
                ('column_with_allowed_values', transform_allowed_values),
                ('columns_to_multiply', transform_multiply),
                ('columns_to_combine', transform_cols_combine),
                ('med_column_to_map', transform_med_map),
                ('column_to_encode', transform_encode),
                ('num_top_features', transform_top_features),
                ('group_by_columns', transform_group_by),
                ('ts_to_summarize_and_regress', transform_timeseries)
            ]

            sheet_df = read_sheet(sheet_config, sheet_name)
            for key, func in transformations:
                if key in sheet_config:
                    sheet_df = func(sheet_df, sheet_config)

            process_function = sheet_processing_map.get(sheet_name)
            if process_function:
                sheet_df = (process_function(sheet_df, sheet_config)
                            if sheet_name in ['dwh Transplantat', 'dwh Diagnose ohne Gruppe']
                            else process_function(sheet_df))

            merged_df = (sheet_df if merged_df is None else
                         (pd.merge(merged_df, sheet_df, on='SPM_ID', how='inner')
                          if sheet_name != 'dwh Transplantat Allokation'
                          else pd.merge(merged_df, sheet_df,
                                        left_on=['SPM_ID', 'dwh.Transplantat.Transplantat'],
                                        right_on=['SPM_ID',
                                                  'dwh.Transplantat.dwh.Transplantat_Allokation.Transplantat'],
                                        how='inner')))

        return merged_df

    def _process_transplantat_sheet(self, sheet_df, sheet_config):
        """Process the transplantat sheet."""
        sheet_df = self.process_target(sheet_df, method=sheet_config['method'])
        return sheet_df
    
    def _merge_excel_files(self, excel_files):
        """Merge all Excel files into a single DataFrame."""
        final_df = None
        for file_config in excel_files:
            file_name = file_config['name']
            sheet_configs = file_config['sheets']
            print('Processing file:', file_name)
            temp_df = self.process_excel_file(os.getenv(file_name), sheet_configs)
            final_df = temp_df if final_df is None else pd.merge(final_df, temp_df, on='SPM_ID', how='inner')
        return final_df

    def drop_rows_with_empty(self, df):
        """Drops all rows that have no Values in all Spender or all Untersuchung Columns (Hinw. Herr Fischer-Fröhlich)"""
        spender_cols = [col for col in df.columns if col.startswith('dwh.Spender.Spender_')]
        if spender_cols:
            ids_to_drop = df.loc[df[spender_cols].isna().all(axis=1), "SPM_ID"]
            df = df.drop(df[df.SPM_ID.isin(ids_to_drop)].index)

        untersuchung_cols = [col for col in df.columns if col.startswith('dwh.Untersuchung_')]
        if untersuchung_cols:
            ids_to_drop = df.loc[df[untersuchung_cols + untersuchung_cols].isna().all(axis=1), "SPM_ID"]
            df = df.drop(df[df.SPM_ID.isin(ids_to_drop)].index)
        return df

    def perform(self, config_path):
        """Execute complete preprocessing pipeline based on configuration."""
        config = self._load_config(config_path)
        config = self._rearrange_config(config, 'Trans_20_24')

        df = self._merge_excel_files(config['excel_files'])

        # Remove date columns and drop rows that are missing specific values based on expert advice
        df = df.drop(columns=[col for col in df.columns if col.endswith("Untersuchungsdatum")])
        df = self.drop_rows_with_empty(df) #TODO: check

        # Save results
        print('Saving Data...')
        dir_path = os.path.join(self.FEATURE_DIR, "dev")
        os.makedirs(dir_path, exist_ok=True)
        clear_csv_files(dir_path)
        dataset_path = os.path.join(dir_path, "merged.pkl")
        df.to_pickle(dataset_path)

if __name__ == "__main__":
    merger = DataMerger()
    merger.perform('src/data/complete_config.yml')
