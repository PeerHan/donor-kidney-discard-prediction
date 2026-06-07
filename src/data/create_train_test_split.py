import os
import random
import pandas as pd
from dotenv import load_dotenv


def get_unique_ids():
    """
    Load a dataset from an Excel file and extract unique 'SPM_ID' values from the 'dwh Transplantat' sheet.

    Returns:
    - unique_ids (DataFrame): A DataFrame containing unique 'SPM_ID' values.
    """
    # Load a dataset
    df = pd.ExcelFile(os.getenv("Trans_20_24"))
    df = pd.read_excel(df, "dwh Transplantat")
    # Filter to get unique 'SPM_ID' values
    unique_ids = pd.DataFrame(df['SPM_ID'].unique(), columns=['SPM_ID'])
    return unique_ids


def drop_ids(df):
    """
    Create a list of IDs to be excluded, then filter out these IDs from the provided DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing 'SPM_ID' to be filtered.

    Returns:
    - filtered_ids (ndarray): An array of 'SPM_ID' values not in the exclusion list.
    """
    # Filter minors
    untersuchung = pd.ExcelFile(os.getenv("Unter_20_24"))
    df_spender = pd.read_excel(untersuchung, 'dwh Spender')
    kinder = df_spender.loc[df_spender.loc[:,'dwh.Spender.Spender_Alter']<18,'SPM_ID']

    # Filter "Nicht an ET gemeldet"
    diagnose = pd.ExcelFile(os.getenv("Diag_20_24"))
    anamnese = pd.read_excel(diagnose, "dwh Anamnesebogen")
    filtered_df = anamnese[anamnese['ET_Registrierung Zeitpunkt'].isna()]
    spm_id_list = filtered_df['SPM_ID'].to_list()

    # Create list of excluded IDs
    dropID = spm_id_list + kinder.to_list()

    #Daten nach Rücksprache mit Herr Fischer-Fröhlich doch im Datensatz lassen
    dropID.remove(12730)
    dropID.remove(14521)

    # save list of excluded IDs
    dropID = pd.DataFrame({"SPM_ID":dropID})
    save_dir = os.getenv("FEATURE_DIR")
    os.makedirs(save_dir, exist_ok=True)
    dropID.to_csv(os.path.join(save_dir, "dropID.csv"), index=False)

    # Filter out 'SPM_ID' values that are in the exclusion list
    filtered_ids = df[~df['SPM_ID'].isin(dropID["SPM_ID"])]['SPM_ID'].to_numpy()
    return filtered_ids


def split(ids, percentage=0.8):
    """
    Randomly split the IDs into training and test sets based on a specified percentage.

    Parameters:
    - ids (ndarray): Array of IDs to be split.
    - percentage (float): Proportion of IDs to be allocated to the training set (default is 0.8).

    Returns:
    - df_train (list): List of training IDs.
    - df_test (list): List of test IDs.
    """
    # Fix random seed
    random.seed(42)
    # Shuffle the IDs
    random.shuffle(ids)
    # Split index based on the specified percentage
    split_point = int(len(ids) * percentage)
    # Split into training and test sets
    df_train = ids[:split_point]
    df_test = ids[split_point:]
    return df_train, df_test


def save(ids_train, ids_test):
    """
    Save the training and test IDs into separate CSV files.

    Parameters:
    - ids_train (list): List of training IDs.
    - ids_test (list): List of test IDs.
    """
    for name, ids in zip(['train', 'test'], [ids_train, ids_test]):
        # Convert list to DataFrame
        df_ids = pd.DataFrame(ids, columns=['SPM_ID'])
        # Save DataFrame to CSV
        save_dir = os.path.join(os.getenv("FEATURE_DIR"), 'train-test-split')
        os.makedirs(save_dir, exist_ok=True)
        df_ids.to_csv(os.path.join(save_dir, f'{name}_ids.csv'), index=False)


def main():
    """
    Main function to orchestrate loading, filtering, splitting, and saving the ID data.
    """
    load_dotenv()
    ids = get_unique_ids()               # Load unique IDs
    ids = drop_ids(ids)                  # Filter out unwanted IDs
    ids_train, ids_test = split(ids)     # Split into train/test sets
    save(ids_train, ids_test)            # Save train/test sets to CSV files


if __name__ == '__main__':
    main()
