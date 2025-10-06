"""
Tests unitarios para la API Flask.
"""

import pytest
import json
import sys
import os

# Agregar directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, load_model


@pytest.fixture
def client():
    """Fixture para crear un cliente de prueba."""
    app.config['TESTING'] = True
    
    # Cargar modelo antes de las pruebas
    load_model()
    
    with app.test_client() as client:
        yield client


def test_health_check(client):
    """Test del endpoint de health check."""
    response = client.get('/')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['status'] == 'healthy'
    assert 'model_info' in data
    assert 'timestamp' in data


def test_predict_valid_input(client):
    """Test de predicción con entrada válida."""
    # Datos de ejemplo (30 características del dataset)
    test_data = {
        "features": [
            17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
            0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
            184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['status'] == 'success'
    assert 'prediction' in data
    assert 'class' in data['prediction']
    assert 'diagnosis' in data['prediction']
    assert 'confidence' in data['prediction']
    assert 'probabilities' in data['prediction']


def test_predict_missing_features(client):
    """Test con campo 'features' faltante."""
    test_data = {
        "data": [1, 2, 3]
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'


def test_predict_wrong_feature_count(client):
    """Test con número incorrecto de características."""
    test_data = {
        "features": [1, 2, 3, 4, 5]  # Solo 5 en lugar de 30
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'


def test_predict_invalid_data_type(client):
    """Test con tipos de datos inválidos."""
    test_data = {
        "features": ["a", "b", "c"] * 10
    }
    
    response = client.post(
        '/predict',
        data=json.dumps(test_data),
        content_type='application/json'
    )
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'


def test_predict_no_json(client):
    """Test sin enviar JSON."""
    response = client.post('/predict')
    
    assert response.status_code == 400
    data = json.loads(response.data)
    assert data['status'] == 'error'


def test_model_info(client):
    """Test del endpoint de información del modelo."""
    response = client.get('/model-info')
    
    assert response.status_code == 200
    data = json.loads(response.data)
    
    assert data['status'] == 'success'
    assert 'model_metadata' in data


def test_404_endpoint(client):
    """Test de endpoint no existente."""
    response = client.get('/nonexistent')
    
    assert response.status_code == 404
    data = json.loads(response.data)
    assert data['status'] == 'error'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
