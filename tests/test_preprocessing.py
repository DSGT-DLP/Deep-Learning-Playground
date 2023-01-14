import pytest
from backend.aws_helpers.lambda_utils.lambda_client import invoke
import json
import pandas as pd

@pytest.fixture
def read_csv_to_json():
    df = pd.read_csv("tests/test_data/iris.csv")
    return df.to_dict("records")
    
@pytest.mark.parametrize(
    "code_snippet",
    [
        """import pandas as pd\ndef preprocess(df):\n\tdf = df.rename(columns={"variety": "dlp"})\n\treturn df""",
    ],
)
def test_rename_column(read_csv_to_json, code_snippet):
    payload = json.dumps({
            "data": read_csv_to_json,
            "code": code_snippet
    })
    resJson = invoke('preprocess_data', payload)
    assert "columns" in resJson and resJson['columns'] == ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'dlp'], resJson

@pytest.mark.parametrize(
    "code_snippet",
    [
        """import pandas as pd\ndef preprocess(df):\n\tdf = df.drop(columns=['variety'])\n\treturn df""",
    ],
)
def test_drop_column(read_csv_to_json, code_snippet):
    payload = json.dumps({
            "data": read_csv_to_json,
            "code": code_snippet
    })
    resJson = invoke('preprocess_data', payload)
    assert "columns" in resJson and resJson['columns'] == ['sepal.length', 'sepal.width', 'petal.length', 'petal.width'], resJson

@pytest.mark.parametrize(
    "code_snippet",
    [
        """import pandas as pd\ndef preprocess(df):\n\tdf['test'] = df['sepal.length'] + df['sepal.width']\n\treturn df""",
    ],
)
def test_add_two_columns(read_csv_to_json, code_snippet):
    payload = json.dumps({
            "data": read_csv_to_json,
            "code": code_snippet
    })
    resJson = invoke('preprocess_data', payload)
    assert "columns" in resJson and resJson['columns'] == ['sepal.length', 'sepal.width', 'petal.length', 'petal.width', 'variety', 'test'], resJson