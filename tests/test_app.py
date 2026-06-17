from app import create_app


def test_home_page_renders():
    app = create_app()
    client = app.test_client()
    response = client.get('/')
    assert response.status_code == 200
    assert b'Livestock Food Prediction' in response.data
    assert b'Afghanistan' in response.data
    assert b'Asses' in response.data


def test_prediction_endpoint_accepts_model_features():
    app = create_app()
    client = app.test_client()
    response = client.post('/predict', json={
        'Area': 'Afghanistan',
        'Item': 'Asses',
        'Element': 'Stocks',
        'Year': 2020,
        'Unit': 'Head',
    })
    assert response.status_code == 200
    data = response.get_json()
    assert data['success'] is True
    assert 'prediction' in data
