import json
import pandas as pd
import numpy as np
import traceback


def preprocess(data, code):
    try:
        df = pd.DataFrame.from_records(data)
        exec_namespace = {}
        exec(code, None, exec_namespace)
        df = exec_namespace["preprocess"](df)
        data = df.to_dict("records")
        columns = pd.Index.tolist(df.columns)
    except Exception:
        return {"statusCode": 400, "message": traceback.format_exc().splitlines()[-1]}
    else:
        return {"statusCode": 200, "data": data, "columns": columns}


def lambda_handler(event, context):
    return preprocess(event["data"], event["code"])
