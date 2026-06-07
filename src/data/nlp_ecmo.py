# ECMO-Extraktion
import os
import re
import pandas as pd
from dotenv import load_dotenv


def preprocess_table(excel_file, sheet_name):
    """
    excel_file: zuvor mit import_file eingelesene Excel-Datei
    sheet_name: Name des Tabellenblatts
    
    Diese Funktion liest ein Excels-Sheet ein und filtert auf die Train-IDs. 
    Anschließend werden die Spaltennamen gekürzt und ein DataFrame zurück gegeben.
    """
    df = pd.read_excel(os.getenv(excel_file), sheet_name )

    # keep only training data
    train_id = pd.read_csv(f'{os.getenv("FEATURE_DIR")}/train-test-split/train_ids.csv')
    df = df.loc[df['SPM_ID'].isin(train_id['SPM_ID'])]

    # remove sheet name from column names
    df.columns = [x.replace('_',' ') for x in df.columns]
    df.columns = [re.sub(r"dwh\.\w+\.", "",x) for x in df.columns]

    return df

load_dotenv()
filtered = preprocess_table('Diag_20_24', 'dwh Anamnesebogen')
filtered_virologie=preprocess_table('Labor_20_24', 'dwh Labor Virologie')


def check_et_bemerkung(text):
    """
    Check if the term 'ECMO' is present in the text.
    """
    if pd.isnull(text) or not text.strip():  # Skip empty or null text
        return "nicht vorhanden"
    text = text.lower()  # Case insensitive
    if "ecmo" in text:
        return "vorhanden"
    return "nicht vorhanden"

# Apply the function to create a new column
filtered['ecmo_status'] = filtered['ET Bemerkung Spender'].apply(check_et_bemerkung)

# Calculate statistics
total_rows = len(filtered)
ecmo_vorhanden = (filtered['ecmo_status'] == "vorhanden").sum()
ecmo_nicht_vorhanden = (filtered['ecmo_status'] == "nicht vorhanden").sum()

# Output results
#print(f"Total rows: {total_rows}")
#print(f"'ECMO' vorhanden: {ecmo_vorhanden}")
#print(f"'ECMO' nicht vorhanden: {ecmo_nicht_vorhanden}")

def check_ecmo(text):
    if pd.isnull(text) or not text.strip():
        return "No Text"
    text = text.lower()
    if "ecmo" in text:
        return "Yes"
    return "No"


filtered['ecmo_mention'] = filtered['ET Bemerkung Spender'].apply(check_ecmo)


total_rows = len(filtered)
ecmo_yes = (filtered["ecmo_mention"] == "Yes").sum()
ecmo_no = (filtered["ecmo_mention"] == "No").sum()
ecmo_no_text = (filtered["ecmo_mention"] == "No Text").sum()

# Output results
#print(f"Total rows: {total_rows}")
#print(f"ECMO mentioned (Yes): {ecmo_yes}")
#print(f"ECMO not mentioned (No): {ecmo_no}")
#print(f"No text (No Text): {ecmo_no_text}")



no_ecmo_texts = filtered[filtered["ecmo_mention"] == "No"]
yes_ecmo_texts = filtered[filtered["ecmo_mention"] == "Yes"]


#print("Texts with 'No' ECMO Mention:")
#print(no_ecmo_texts[["SPM ID", "ET Bemerkung Spender"]])

#print("\nTexts with 'Yes' ECMO Mention:")
#print(yes_ecmo_texts[["SPM ID", "ET Bemerkung Spender"]])




# Validate results using ECMO Therapie in Untersuchung/Monitoring 
df_validation = preprocess_table('Unter_20_24', 'dwh Monitoring')
df_validation_unique = df_validation.groupby("SPM ID").agg(
    {"ECMO Therapie": lambda x: x.dropna().iloc[0] if not x.dropna().empty else pd.NA}
).reset_index()

# Found in Both: ECMO is mentioned in both columns
# Only in ET Bem Spender:
#       ECMO mentioned in ET Bemerkung Spender,
#       but missing in Ecmo THerapie
# Only in Untersuchung/Monitoring:
#       ECMO mentioned in ECMO Therapie,
#       but missing in Spalte ET Bemerkung Spender
# Not found: ECMO not mentioned in either column


merged = pd.merge(
    filtered[["SPM ID", "ET Bemerkung Spender"]],
    df_validation_unique[["SPM ID", "ECMO Therapie"]],
    on="SPM ID",
    how="left",
)

merged["ecmo_mention"] = merged["ET Bemerkung Spender"].apply(check_ecmo)

merged["validation_status"] = merged.apply(
    lambda row: "Found in Both" if row["ecmo_mention"] == "Yes" and pd.notnull(row["ECMO Therapie"])
    else "Only in ET Bem Spender" if row["ecmo_mention"] == "Yes" and pd.isnull(row["ECMO Therapie"])
    else "Only in Untersuchung/Monitoring" if row["ecmo_mention"] == "No" and pd.notnull(row["ECMO Therapie"])
    else "Not Found",
    axis=1,
)


result = merged[["SPM ID", "ET Bemerkung Spender", "validation_status"]]

correct_results = result[result["validation_status"] == "Found in Both"]

total_rows = len(result)
found_in_both = (result["validation_status"] == "Found in Both").sum()
only_in_ecmo_mention = (result["validation_status"] == "Only in ET Bem Spender").sum()
only_in_ecmo_therapy = (result["validation_status"] == "Only in Untersuchung/Monitoring").sum()
not_found = (result["validation_status"] == "Not Found").sum()

# Output results
#print(f"Total rows: {total_rows}")
#print(f"Found in Both: {found_in_both}")
#print(f"Only in ET Bem Spender: {only_in_ecmo_mention}")
#print(f"Only in Untersuchung/Monitoring: {only_in_ecmo_therapy}")
#print(f"Not Found: {not_found}")

# Anzahl der ECMO Erwähnungen(unique SPM_ID)
only_et = result[result["validation_status"] == "Only in ET Bem Spender"]

# Save for use in imputation
only_et.to_csv(f"{os.getenv('FEATURE_DIR')}/fill_ecmo_ids.csv")

# Display the texts
#print("Texts with 'Only in ET Bem Spender' validation status:")
#print(only_et)
