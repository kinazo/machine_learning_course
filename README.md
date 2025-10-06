# machine_learning_course
Proyectos realizados durante mi curso de especialización en Machine Learning

# Breast Cancer Prediction API - MLOps Pipeline

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Sistema completo de MLOps para predicción de cáncer de mama usando Machine Learning, desplegado como API REST con containerización Docker y pipeline CI/CD automatizado.**

---

## Descripción

Este proyecto implementa un **pipeline MLOps completo** para la clasificación de tumores de mama como benignos o malignos utilizando el dataset **Breast Cancer Wisconsin (Diagnostic)**. El sistema incluye:

- **Modelo de Machine Learning**: Random Forest con optimización de hiperparámetros
- **API REST**: Servicio Flask para inferencia en tiempo real
- **Containerización**: Docker para despliegue reproducible
- **CI/CD**: Pipeline automatizado con GitHub Actions
- **Testing**: Suite completa de pruebas unitarias
- **Monitoreo**: Health checks y logging estructurado

**Dataset**: 30 características calculadas a partir de imágenes digitalizadas de biopsias de masa mamaria.

---

## Características

### Machine Learning
- Modelo Random Forest con **95%+ de precisión** (AUC-ROC)
- Optimización automática de hiperparámetros con GridSearchCV
- Normalización de features con StandardScaler
- Validación cruzada 5-fold
- Análisis de importancia de características

### API REST
- Endpoints para predicción, health check e información del modelo
- Validación robusta de entradas
- Manejo de errores y respuestas estructuradas
- Logging detallado de requests
- Documentación completa de API

### DevOps
- Containerización con Docker multi-stage
- Docker Compose para orquestación
- Pipeline CI/CD con GitHub Actions
- Tests automatizados en cada commit
- Health checks integrados
- Usuario no-root para seguridad

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                    GitHub                            │
│              (Código + CI/CD)                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              GitHub Actions                          │
│         (Build, Test, Deploy)                        │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│              Docker Image                            │
│          (breast-cancer-api)                         │
└──────────────────┬──────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────┐
│                                                      │
│  ┌────────────────────────────────────────────┐    │
│  │         Flask API (Port 5000)              │    │
│  │                                             │    │
│  │  ┌──────────────┐  ┌──────────────┐       │    │
│  │  │  Health      │  │   Predict    │       │    │
│  │  │  Check       │  │   Endpoint   │       │    │
│  │  └──────┬───────┘  └──────┬───────┘       │    │
│  │         │                  │               │    │
│  │         └──────────┬───────┘               │    │
│  │                    ▼                       │    │
│  │         ┌─────────────────────┐            │    │
│  │         │   Random Forest     │            │    │
│  │         │   + Scaler          │            │    │
│  │         └─────────────────────┘            │    │
│  └────────────────────────────────────────────┘    │
│                                                      │
│              Docker Container                        │
└─────────────────────────────────────────────────────┘
```

---

## Tecnologías

### Machine Learning & Data Science
- **Python 3.9+**
- **scikit-learn 1.3.2** - Modelo y preprocesamiento
- **NumPy 1.24.3** - Operaciones numéricas
- **Pandas 2.1.4** - Manipulación de datos
- **Joblib 1.3.2** - Serialización de modelos

### Backend & API
- **Flask 3.0.0** - Framework web
- **Werkzeug 3.0.1** - Utilidades WSGI

### DevOps & Testing
- **Docker** - Containerización
- **Docker Compose** - Orquestación
- **pytest** - Testing framework
- **GitHub Actions** - CI/CD

---

## Instalación

### Prerrequisitos

- Python 3.9 o superior
- pip
- Docker (opcional)
- Git

### Opción 1: Instalación Local

```bash
# Clonar el repositorio
git clone https://github.com/tu-usuario/breast-cancer-mlops.git
cd breast-cancer-mlops

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Linux/Mac:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Entrenar el modelo
python train_model.py

# Ejecutar la API
python app.py
```

### Opción 2: Usando Make

```bash
# Instalar dependencias
make install

# Entrenar modelo
make train

# Ejecutar tests
make test

# Ejecutar API
make run
```

### Opción 3: Docker (Recomendado)

```bash
# Construir imagen
docker build -t breast-cancer-api:latest .

# Ejecutar contenedor
docker run -d -p 5000:5000 --name bc-api breast-cancer-api:latest

# Ver logs
docker logs -f bc-api
```

### Opción 4: Docker Compose

```bash
# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

---

## Uso

### 1. Verificar Estado del Servicio

```bash
curl http://localhost:5000/
```

**Respuesta:**
```json
{
  "status": "healthy",
  "message": "Servicio de predicción de cáncer de mama operativo",
  "model_info": {
    "trained_at": "2024-01-15T10:30:00",
    "test_auc": 0.9876,
    "features_count": 30
  },
  "timestamp": "2024-01-15T15:45:00"
}
```

### 2. Realizar Predicción

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [
      17.99, 10.38, 122.8, 1001, 0.1184, 0.2776, 0.3001, 0.1471,
      0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.4, 0.006399,
      0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33,
      184.6, 2019, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189
    ]
  }'
```

**Respuesta:**
```json
{
  "status": "success",
  "prediction": {
    "class": 0,
    "diagnosis": "Maligno",
    "confidence": 95.67,
    "probabilities": {
      "malignant": 95.67,
      "benign": 4.33
    }
  },
  "timestamp": "2024-01-15T15:45:00"
}
```

### 3. Pruebas Automatizadas

```bash
# Ejecutar script de pruebas
python test_requests.py http://localhost:5000
```

---

## API Endpoints

### `GET /`
**Health Check** - Verifica el estado del servicio

**Response:**
- `200 OK` - Servicio operativo con información del modelo
- `503 Service Unavailable` - Modelo no cargado

---

### `POST /predict`
**Predicción** - Clasifica un tumor como benigno o maligno

**Request Body:**
```json
{
  "features": [float, float, ..., float]  // 30 valores numéricos
}
```

**Response:**
- `200 OK` - Predicción exitosa
- `400 Bad Request` - Datos inválidos o incompletos
- `503 Service Unavailable` - Modelo no disponible

**Ejemplo de respuesta exitosa:**
```json
{
  "status": "success",
  "prediction": {
    "class": 1,
    "diagnosis": "Benigno",
    "confidence": 98.45,
    "probabilities": {
      "malignant": 1.55,
      "benign": 98.45
    }
  },
  "timestamp": "2024-01-15T15:45:00"
}
```

---

### `GET /model-info`
**Información del Modelo** - Retorna metadata completa del modelo

**Response:**
```json
{
  "status": "success",
  "model_metadata": {
    "trained_at": "2024-01-15T10:30:00",
    "test_auc": 0.9876,
    "best_params": {
      "n_estimators": 200,
      "max_depth": 20,
      "min_samples_split": 2,
      "min_samples_leaf": 1
    },
    "top_features": [
      {"feature": "worst perimeter", "importance": 0.145},
      {"feature": "worst radius", "importance": 0.132}
    ]
  }
}
```

---

##  Docker

### Construir Imagen

```bash
# Build básico
docker build -t breast-cancer-api:latest .

# Build con tag específico
docker build -t breast-cancer-api:v1.0.0 .

# Verificar imagen
docker images | grep breast-cancer-api
```

### Ejecutar Contenedor

```bash
# Ejecución básica
docker run -d -p 5000:5000 --name bc-api breast-cancer-api:latest

# Con variables de entorno
docker run -d \
  -p 5000:5000 \
  -e FLASK_ENV=production \
  --name bc-api \
  --restart unless-stopped \
  breast-cancer-api:latest

# Ver logs
docker logs -f bc-api

# Entrar al contenedor
docker exec -it bc-api /bin/bash

# Detener y eliminar
docker stop bc-api
docker rm bc-api
```

### Docker Compose

```bash
# Iniciar servicios en background
docker-compose up -d

# Ver logs en tiempo real
docker-compose logs -f api

# Reiniciar servicios
docker-compose restart

# Detener y limpiar
docker-compose down -v
```

---

##  CI/CD

El proyecto incluye un pipeline completo en **GitHub Actions** que se ejecuta en cada push y pull request.

### Pipeline Stages

1. ** Test Stage**
   - Instala dependencias de Python
   - Verifica estructura del proyecto
   - Entrena el modelo (si no existe)
   - Ejecuta suite de tests unitarios
   - Genera reporte de cobertura

2. ** Build Stage**
   - Construye imagen Docker
   - Ejecuta healthcheck del contenedor
   - Valida funcionamiento de la API
   - Guarda imagen como artefacto

3. ** Deploy Stage** (Opcional)
   - Push a Docker Hub
   - Tag con SHA del commit
   - Deploy automático

---

##  Testing

### Ejecutar Tests

```bash
# Tests básicos
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=. --cov-report=html

# Tests específicos
pytest tests/test_api.py::test_health_check -v

# Con make
make test
```

### Tests Incluidos

-  **test_health_check** - Verifica endpoint de salud
-  **test_prediction_valid** - Predicción con datos válidos
-  **test_prediction_invalid_features** - Validación de datos incorrectos
-  **test_prediction_missing_features** - Validación de campos faltantes
-  **test_model_info** - Endpoint de información del modelo

### Cobertura de Tests

Los tests cubren:
- Carga del modelo
- Validación de entradas
- Endpoints de la API
- Manejo de errores
- Respuestas JSON

---

##  Modelo ML

### Detalles del Modelo

- **Algoritmo**: Random Forest Classifier
- **Framework**: scikit-learn 1.3.2
- **Dataset**: Breast Cancer Wisconsin (Diagnostic)
  - 569 muestras
  - 30 características numéricas
  - 2 clases (Maligno/Benigno)

### Métricas de Performance

| Métrica | Valor |
|---------|-------|
| **AUC-ROC** | ~0.98 |
| **Precisión** | ~96% |
| **Recall** | ~95% |
| **F1-Score** | ~95% |

### Características del Dataset

Las 30 características se dividen en 3 grupos (mean, error estándar, worst):

1. **Radio** - Radio medio de núcleos celulares
2. **Textura** - Desviación estándar de valores en escala de grises
3. **Perímetro** - Perímetro del núcleo
4. **Área** - Área del núcleo
5. **Suavidad** - Variación local en longitudes de radio
6. **Compacidad** - (perímetro² / área) - 1.0
7. **Concavidad** - Severidad de porciones cóncavas del contorno
8. **Puntos cóncavos** - Número de porciones cóncavas del contorno
9. **Simetría** - Simetría del núcleo
10. **Dimensión fractal** - "aproximación de línea costera" - 1

Cada característica tiene 3 versiones: **mean**, **standard error**, **worst** (promedio de los 3 valores más grandes).

### Top 5 Características Más Importantes

1. **Worst Perimeter** (~14.5%)
2. **Worst Radius** (~13.2%)
3. **Mean Concave Points** (~12.8%)
4. **Worst Concave Points** (~11.9%)
5. **Mean Perimeter** (~10.3%)

---

##  Estructura del Proyecto

```
breast-cancer-mlops/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml              # Pipeline CI/CD
│
├── model/
│   ├── breast_cancer_model.pkl   # Modelo entrenado + scaler
│   └── model_metadata.json       # Metadata del modelo
│
├── tests/
│   ├── __init__.py
│   └── test_api.py               # Tests unitarios
│
├── app.py                         # API Flask principal
├── train_model.py                 # Script de entrenamiento
├── test_requests.py               # Tests manuales de la API
│
├── requirements.txt               # Dependencias Python
├── Dockerfile                     # Configuración Docker
├── docker-compose.yml             # Orquestación Docker
├── Makefile                       # Comandos útiles
│
├── .gitignore
├── .dockerignore
├── README.md                      # Este archivo
└── LICENSE                        # Licencia del proyecto
```

---

### Guidelines

- Asegúrate de que los tests pasen (`make test`)
- Añade tests para nuevas funcionalidades
- Actualiza la documentación según sea necesario
- Sigue las convenciones de código del proyecto

---

##  Roadmap

- [ ] Integración con bases de datos (PostgreSQL)
- [ ] Sistema de logs centralizado (ELK Stack)
- [ ] Monitoreo con Prometheus + Grafana
- [ ] API authentication con JWT
- [ ] Rate limiting
- [ ] Soporte para múltiples modelos
- [ ] Frontend web interactivo
- [ ] Deploy en Kubernetes
- [ ] Documentación con Swagger/OpenAPI
- [ ] Model versioning con MLflow

---

## Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

##  Agradecimientos

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- scikit-learn Documentation
- Flask Documentation
- Docker Documentation

---

##  Referencias

1. Wolberg, W., Street, W., & Mangasarian, O. (1995). Breast Cancer Wisconsin (Diagnostic) Data Set.
2. Sklearn Random Forest Classifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
3. Flask Documentation: https://flask.palletsprojects.com/
4. Docker Best Practices: https://docs.docker.com/develop/dev-best-practices/

---
