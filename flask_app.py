"""
API REST para servir predicciones del modelo de cáncer de mama.
Expone endpoints para verificación de estado y predicciones.
"""

from flask import Flask, request, jsonify
import joblib
import numpy as np
import logging
from datetime import datetime
import traceback
import os

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Inicializar Flask
app = Flask(__name__)

# Variables globales para el modelo
model_artifacts = None
MODEL_PATH = 'model/breast_cancer_model.pkl'


def load_model():
    """Carga el modelo y sus artefactos al inicio."""
    global model_artifacts
    
    try:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}")
        
        model_artifacts = joblib.load(MODEL_PATH)
        logger.info("Modelo cargado exitosamente!")
        logger.info(f"Características esperadas: {len(model_artifacts['metadata']['feature_names'])}")
        
        return True
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return False


def validate_input(data):
    """
    Valida que los datos de entrada sean correctos.
    
    Args:
        data: Diccionario con las características
        
    Returns:
        tuple: (is_valid, error_message, processed_features)
    """
    if not data:
        return False, "No se proporcionaron datos", None
    
    if 'features' not in data:
        return False, "Campo 'features' requerido", None
    
    features = data['features']
    
    if not isinstance(features, list):
        return False, "Las características deben ser una lista", None
    
    expected_features = len(model_artifacts['metadata']['feature_names'])
    
    if len(features) != expected_features:
        return False, f"Se esperan {expected_features} características, se recibieron {len(features)}", None
    
    # Validar que todos sean numéricos
    try:
        features_array = np.array(features, dtype=float).reshape(1, -1)
    except (ValueError, TypeError):
        return False, "Todas las características deben ser valores numéricos", None
    
    # Validar que no haya valores NaN o infinitos
    if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
        return False, "Las características contienen valores inválidos (NaN o infinito)", None
    
    return True, None, features_array


@app.route('/', methods=['GET'])
def health_check():
    """
    Endpoint de verificación de estado del servicio.
    
    Returns:
        JSON con estado del servicio y metadata del modelo
    """
    try:
        if model_artifacts is None:
            return jsonify({
                'status': 'error',
                'message': 'Modelo no cargado',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        response = {
            'status': 'healthy',
            'message': 'Servicio de predicción de cáncer de mama operativo',
            'model_info': {
                'trained_at': model_artifacts['metadata'].get('trained_at', 'unknown'),
                'test_auc': model_artifacts['metadata'].get('test_auc', 'unknown'),
                'features_count': len(model_artifacts['metadata']['feature_names'])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info("Health check exitoso")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en health check: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error interno del servidor',
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para realizar predicciones.
    
    Espera un JSON con el formato:
    {
        "features": [val1, val2, ..., val30]
    }
    
    Returns:
        JSON con la predicción y probabilidades
    """
    try:
        # Verificar que el modelo esté cargado
        if model_artifacts is None:
            logger.error("Intento de predicción sin modelo cargado")
            return jsonify({
                'status': 'error',
                'message': 'Modelo no disponible',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Obtener datos del request
        data = request.get_json()
        
        if data is None:
            return jsonify({
                'status': 'error',
                'message': 'Request debe ser JSON válido',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Validar entrada
        is_valid, error_msg, features_array = validate_input(data)
        
        if not is_valid:
            logger.warning(f"Validación fallida: {error_msg}")
            return jsonify({
                'status': 'error',
                'message': error_msg,
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Escalar características
        features_scaled = model_artifacts['scaler'].transform(features_array)
        
        # Realizar predicción
        prediction = int(model_artifacts['model'].predict(features_scaled)[0])
        probabilities = model_artifacts['model'].predict_proba(features_scaled)[0]
        
        # Interpretar resultado
        diagnosis = "Benigno" if prediction == 1 else "Maligno"
        confidence = float(max(probabilities)) * 100
        
        response = {
            'status': 'success',
            'prediction': {
                'class': prediction,
                'diagnosis': diagnosis,
                'confidence': round(confidence, 2),
                'probabilities': {
                    'malignant': round(float(probabilities[0]) * 100, 2),
                    'benign': round(float(probabilities[1]) * 100, 2)
                }
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Predicción exitosa: {diagnosis} ({confidence:.2f}%)")
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'status': 'error',
            'message': 'Error procesando la predicción',
            'detail': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    """
    Endpoint para obtener información detallada del modelo.
    
    Returns:
        JSON con metadata completa del modelo
    """
    try:
        if model_artifacts is None:
            return jsonify({
                'status': 'error',
                'message': 'Modelo no cargado'
            }), 503
        
        metadata = model_artifacts['metadata'].copy()
        
        response = {
            'status': 'success',
            'model_metadata': metadata,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error obteniendo info del modelo: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Error obteniendo información del modelo'
        }), 500


@app.errorhandler(404)
def not_found(e):
    """Manejo de rutas no encontradas."""
    return jsonify({
        'status': 'error',
        'message': 'Endpoint no encontrado',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Manejo de errores internos."""
    return jsonify({
        'status': 'error',
        'message': 'Error interno del servidor',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    # Cargar modelo al inicio
    if load_model():
        logger.info("Iniciando servidor Flask...")
        app.run(host='0.0.0.0', port=5000, debug=False)
    else:
        logger.error("No se pudo cargar el modelo. Servidor no iniciado.")
        exit(1)
