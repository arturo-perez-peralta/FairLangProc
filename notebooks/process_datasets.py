import json, argparse
import pandas as pd
from pathlib import Path

# --- Description functions ---
def describe_german(row):
    gender, marital = ("", None)
    if isinstance(row["personal_status"], list) and len(row["personal_status"]) == 2:
        gender, marital = row["personal_status"]
        gender = " " + gender
    elif isinstance(row["personal_status"], tuple):
         gender, marital = row["personal_status"]
         gender = " " + gender


    desc = f"A {int(row['age'])}-year-old{gender} applicant who"
    if marital:
        desc += f" is {marital} and"

    if isinstance(row["employment"], str) and "unemployed" not in row["employment"]:
        desc += f" has been {row['employment']}"
    elif "unemployed" in str(row["employment"]):
        desc += " is currently unemployed"
    desc += "."

    desc += (
        f" They applied for a loan of {int(row['credit_amount'])} Deutsche Marks "
        f"to finance {row['purpose']} lasting {int(row['duration'])} months."
    )
    desc += (
        f" Their checking account balance is {row['checking_status']} "
        f"and they have savings of {row['savings_status']}."
    )
    desc += f" Their credit history shows {row['credit_history']}."
    desc += f" They {('own' if row['housing'] == 'owned' else 'rent')} their home and possess {row['property_magnitude']}."
    if row["existing_credits"] > 1:
        desc += f" They currently hold {int(row['existing_credits'])} existing credits."
    else:
        desc += " They currently hold only one existing credit."
    if row["num_dependents"] > 0:
        desc += f" They support {int(row['num_dependents'])} dependent(s)."
    if row["own_telephone"] == "yes":
        desc += " They have a telephone registered in their name."
    else:
        desc += " They do not have a personal telephone."
    if row["foreign_worker"] == "yes":
        desc += " The applicant is a foreign worker."
    else:
        desc += " The applicant is not a foreign worker."
    return desc


def describe_adult(row):
    desc = f"A {int(row['age'])}-year-old {row['sex'].lower()}"
    if isinstance(row["marital_status"], str):
        desc += f" who is {row['marital_status'].replace('-', ' ').lower()}"
    if isinstance(row["education"], str):
        desc += f" and has an education level of {row['education'].lower()}"
    desc += "."
    if isinstance(row["occupation"], str):
        desc += f" They work as a {row['occupation'].replace('-', ' ').lower()}"
    if isinstance(row["workclass"], str):
        desc += f" in the {row['workclass'].replace('-', ' ').lower()} sector"
    desc += f", typically working {int(row['hours_per_week'])} hours per week."
    desc += f" Their race is {row['race'].lower()}, and they are from {row['native_country']}."
    return desc


def describe_compas(row):
    desc = f"A {int(row['age'])}-year-old {row['sex'].lower()} of {row['race'].lower()} race"

    # Type of current charge
    if 'c_charge_degree' in row and pd.notna(row['c_charge_degree']):
        charge_type = str(row['c_charge_degree'])
        if charge_type == 'F':
            desc += " currently facing a felony charge"
        elif charge_type == 'M':
            desc += " currently facing a misdemeanor charge"
        else:
            desc += f" currently facing a charge of type {charge_type}"
    desc += "."

    # Number of prior offenses
    if 'priors_count' in row and pd.notna(row['priors_count']):
        priors = int(row['priors_count'])
        if priors == 0:
            desc += " They have no prior offenses."
        elif priors == 1:
            desc += " They have one prior offense."
        else:
            desc += f" They have {priors} prior offenses."

    # Juvenile history
    juv_vars = ['juv_fel_count', 'juv_misd_count', 'juv_other_count']
    juv_desc = []
    for col in juv_vars:
        if col in row and pd.notna(row[col]) and int(row[col]) > 0:
            juv_desc.append(f"{int(row[col])} {col.replace('_', ' ').replace('count', '')}")
    if juv_desc:
        desc += " Their juvenile record includes: " + ", ".join(juv_desc) + "."

    return desc

# --- Processing logic ---
def load_config(config_path="config_datasets.json"):
    """Carga el archivo de configuración JSON."""
    with open(config_path, "r") as f:
        return json.load(f)

def process(dataset_name: str, config: dict):
    """
    Procesa un dataset específico basado en la configuración proporcionada.
    """
    if dataset_name not in config:
        raise ValueError(f"No se encontró configuración para el dataset: {dataset_name}")

    d_config = config[dataset_name]
    
    path = Path("datasets").absolute() / dataset_name / d_config["csv_file"]
    sep = d_config["separator"]
    columns = d_config.get("columns")
    mappings = d_config.get("mappings")

    if columns:
        df = pd.read_csv(path, sep=sep, names=columns)
        df.dropna(inplace=True)
    else:
        df = pd.read_csv(path, sep=sep)
        
    if mappings:
        for col, mapping in mappings.items():
            df[col] = df[col].map(mapping).fillna(df[col])
            
    # Registro de funciones de descripción
    desc_fun = {
        'german': describe_german,
        'adult': describe_adult,
        'compas': describe_compas,
    }
    
    if dataset_name not in desc_fun:
         raise ValueError(f"No se encontró función de descripción para el dataset: {dataset_name}")

    df["description"] = df.apply(desc_fun[dataset_name], axis=1)

    return df

def main():
    """
    Main function for parsing arguments and executing processing.
    """
    parser = argparse.ArgumentParser(description="Process tabular datasets to textual descriptions.")
    parser.add_argument("dataset", 
                        choices=["german", "adult", "compas"], 
                        help="Dataset name to process.")
    parser.add_argument("--config", 
                        default="config_datasets.json", 
                        help="JSON rute to config file.")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"Processing dataset: {args.dataset}...")
        df = process(args.dataset, config)
        
        output_filename = f"{args.dataset}_processed.csv"
        df.to_csv(output_filename, index=False)
        print("Processing complete. Results saved to:", output_filename)
        print(f"--- 5 first rows sample ---")
        print(df.head())
        print("\n--- Description sample ---")
        print(df.iloc[0]["description"])

    except Exception as e:
        print(f"Processing error: {e}")

if __name__ == "__main__":
    main()