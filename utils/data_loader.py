import pandas as pd

def load_csv_safe(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin1")
