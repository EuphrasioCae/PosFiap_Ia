import os
import pandas as pd

def file_exists(path):
    return os.path.exists(path)


def save_if_different(new_df, file_path):
    if not file_exists(file_path):
        new_df.to_parquet(file_path, index=False)
        print("Arquivo salvo com sucesso")
        return
    
    old_df = pd.read_parquet(file_path)
    
    if new_df.equals(old_df):
        print("Arquivo em sua última versão!")
    else:
        new_df.to_parquet(file_path, index=False)
        print("Arquivo atualizado com sucesso")


def load_parquet_safe(file_path, notebook_origem):
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Erro: O arquivo '{file_path}' está faltando. "
            f"Por favor, execute o notebook '{notebook_origem}' antes de continuar."
        )
    return pd.read_parquet(file_path)