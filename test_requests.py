"""
Script para probar manualmente la API con requests.
Útil para verificar que la API funciona correctamente.
"""

import requests
import json
import sys


def test_health_check(base_url):
    """Prueba el endpoint de health check."""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("✓ Health check exitoso")
            return True
        else:
            print("✗ Health check falló")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_prediction(base_url):
    """Prueba el endpoint de predicción con datos válidos."""
    print("\n" + "="*60)
    print("TEST 2: Predicción con datos válidos")
    print("="*60)
    
    # Datos de ejemplo (primer caso del dataset - maligno)
    test_data = {
        "features": [
            17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
            0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
            0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
            184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Predicción exitosa")
            print(f"  Diagnóstico: {result['prediction']['diagnosis']}")
            print(f"  Confianza: {result['prediction']['confidence']}%")
            return True
        else:
            print("✗ Predicción falló")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_prediction_benign(base_url):
    """Prueba con un caso que debería ser benigno."""
    print("\n" + "="*60)
    print("TEST 3: Predicción caso benigno")
    print("="*60)
    
    # Caso típicamente benigno (valores más bajos)
    test_data = {
        "features": [
            13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781,
            0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56, 0.008462,
            0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26,
            99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259
        ]
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"\n✓ Predicción exitosa")
            print(f"  Diagnóstico: {result['prediction']['diagnosis']}")
            print(f"  Confianza: {result['prediction']['confidence']}%")
            return True
        else:
            print("✗ Predicción falló")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_invalid_input(base_url):
    """Prueba con datos inválidos para verificar validación."""
    print("\n" + "="*60)
    print("TEST 4: Validación con datos inválidos")
    print("="*60)
    
    test_data = {
        "features": [1, 2, 3]  # Muy pocas características
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response:\n{json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 400:
            print("✓ Validación funcionando correctamente")
            return True
        else:
            print("✗ Validación no funciona como esperado")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def test_model_info(base_url):
    """Prueba el endpoint de información del modelo."""
    print("\n" + "="*60)
    print("TEST 5: Información del modelo")
    print("="*60)
    
    try:
        response = requests.get(f"{base_url}/model-info")
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            metadata = result.get('model_metadata', {})
            
            print(f"\nMetadata del modelo:")
            print(f"  Entrenado: {metadata.get('trained_at', 'N/A')}")
            print(f"  AUC Test: {metadata.get('test_auc', 'N/A')}")
            print(f"  Mejores parámetros: {metadata.get('best_params', 'N/A')}")
            
            if 'top_features' in metadata:
                print(f"\n  Top 5 características importantes:")
                for feat in metadata['top_features'][:5]:
                    print(f"    - {feat['feature']}: {feat['importance']:.4f}")
            
            print("\n✓ Información del modelo obtenida")
            return True
        else:
            print("✗ No se pudo obtener información del modelo")
            return False
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return False


def main():
    """Función principal para ejecutar todas las pruebas."""
    
    # URL base de la API
    if len(sys.argv) > 1:
        base_url = sys.argv[1]
    else:
        base_url = "http://localhost:5000"
    
    print("="*60)
    print(f"PRUEBAS DE LA API: {base_url}")
    print("="*60)
    
    # Ejecutar todas las pruebas
    tests = [
        test_health_check,
        test_prediction,
        test_prediction_benign,
        test_invalid_input,
        test_model_info
    ]
    
    results = []
    for test in tests:
        try:
            results.append(test(base_url))
        except Exception as e:
            print(f"\nError ejecutando {test.__name__}: {str(e)}")
            results.append(False)
    
    # Resumen
    print("\n" + "="*60)
    print("RESUMEN DE PRUEBAS")
    print("="*60)
    passed = sum(results)
    total = len(results)
    print(f"Pruebas pasadas: {passed}/{total}")
    
    if passed == total:
        print("✓ TODAS LAS PRUEBAS EXITOSAS")
        return 0
    else:
        print("✗ ALGUNAS PRUEBAS FALLARON")
        return 1


if __name__ == "__main__":
    exit(main())
