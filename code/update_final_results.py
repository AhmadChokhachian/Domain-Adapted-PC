import pandas as pd
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

def update_final_results(method, table_id, rmse=np.nan, nlpd=np.nan, runtime=np.nan,
                         final_csv=None):

    if final_csv is None:
        final_csv = REPO_ROOT / "results" / "final results.csv"
    else:
        final_csv = Path(final_csv)

    if not final_csv.exists():
        raise FileNotFoundError(f"{final_csv} does not exist. Please create the template first.")

    df = pd.read_csv(final_csv, header=None, dtype=str).fillna("")

    def find_row(table_label, method_name):
        table_rows = df.index[df[0] == table_label].tolist()
        if not table_rows:
            raise ValueError(f"Could not find {table_label}")
        table_row = table_rows[0]

        next_table_rows = df.index[
            (df[0].isin(["Table 2", "Table 3", "Table 4"])) & (df.index > table_row)
        ].tolist()
        end_row = len(df) - 1 if not next_table_rows else next_table_rows[0] - 1

        method_rows = df.index[df[0] == method_name].tolist()
        method_rows = [r for r in method_rows if table_row < r <= end_row]
        if not method_rows:
            raise ValueError(f"Could not find method {method_name} under {table_label}")
        return method_rows[0]

    r = find_row(table_id, method)

    if table_id in ["Table 2", "Table 3"]:
        df.iat[r, 1] = "" if pd.isna(rmse) else f"{rmse:.2f}"
        df.iat[r, 2] = "" if pd.isna(nlpd) else f"{nlpd:.2f}"
        df.iat[r, 3] = "" if pd.isna(runtime) else f"{runtime:.2f}"
    elif table_id == "Table 4":
        df.iat[r, 1] = "" if pd.isna(rmse) else f"{rmse:.2f}"
    else:
        raise ValueError("table_id must be one of: Table 2, Table 3, Table 4")

    df.to_csv(final_csv, header=False, index=False)