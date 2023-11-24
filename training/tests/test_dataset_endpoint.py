import pytest
from tests.utils.test_utils import mock_authenticate, get_test_bearer_token
from django.test import Client
from training.core.authenticator import FirebaseAuth

@pytest.mark.parametrize(
    "dataset_name", [
        "IRIS",
        "CALIFORNIA_HOUSING",
        "DIABETES",
        "WINE"
    ]
)
def test_columns_endpoint(monkeypatch, dataset_name):
    client = Client()
    # Use monkeypatch to replace FirebaseAuth.authenticate with our mock function
    monkeypatch.setattr(FirebaseAuth, "authenticate", mock_authenticate)
    
    # Set the Authorization header with the fake token
    headers = get_test_bearer_token()
    response = client.get(f'/api/datasets/default/{dataset_name}/columns', **headers)
    assert response.status_code == 200
    assert isinstance(response.json(), dict)

@pytest.mark.parametrize(
    "dataset_name", [
        "TEST_DATASET",
        "HELLO",
        None
    ]
)
def test_columns_invalid_default(monkeypatch, dataset_name):
    client = Client()
    # Use monkeypatch to replace FirebaseAuth.authenticate with our mock function
    monkeypatch.setattr(FirebaseAuth, "authenticate", mock_authenticate)
    headers = get_test_bearer_token() 
    response = client.get(f'/api/datasets/default/{dataset_name}/columns', **headers)
    print(f'Response: {vars(response)}')
    assert response.status_code == 404
    assert response.content.decode('utf-8') == '{"message": "Dataset not found"}'