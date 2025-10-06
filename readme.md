#  API de Predicción de Cáncer de Mama - MLOps Pipeline

Sistema completo de Machine Learning para predicción de cáncer de mama, desplegado como API REST con Docker y CI/CD automatizado.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Flask](https://img.shields.io/badge/Flask-3.0-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-orange)

##  Tabla de Contenidos

- [Descripción del Proyecto](#descripción-del-proyecto)
- [Arquitectura del Sistema](#arquitectura-del-sistema)
- [Características](#características)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Endpoints de la API](#endpoints-de-la-api)
- [Docker](#docker)
- [CI/CD Pipeline](#cicd-pipeline)
- [Testing](#testing)
- [Modelo de Machine Learning](#modelo-de-machine-learning)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Contribución](#contribución)

##  Descripción del Proyecto

Este proyecto implementa un sistema MLOps completo que incluye:

- **Modelo de ML**: Random Forest para clasificación de tumores benignos/malignos
- **API REST**: Servicio Flask para inferencia en tiempo real
- **Containerización**: Docker para despliegue reproducible
- **CI/CD**: GitHub Actions para automatización de pruebas y despliegues
- **Testing**: Suite completa de pruebas unitarias
- **Logging**: Sistema de registro de eventos y errores

**Dataset**: Breast Cancer Wisconsin (Diagnostic) - 30 características calculadas a partir de imágenes de biopsias.

##  Arquitectura del Sistema

```
┌─────────────────┐
│   Usuario/App   │
└────────┬────────┘
         │ HTTP Request
         ▼
┌─────────────────┐
│   Flask API     │
│   (Puerto 5000) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Modelo ML      │
│  Random Forest  │
└─────────────────┘
```

##  Características

-  Modelo de clasificación con 95%+ de precisión (AUC-ROC)
-  API REST con validación de entradas
-  Manejo robusto de errores y logging
-  Contenedorización con Docker
-  Pipeline CI/CD automatizado
-  Tests unitarios automatizados
-  Documentación completa
-  Health checks y monitoreo
-  Escalabilidad mediante contenedores

##  Requisitos

### Desarrollo Local
- Python 3.9+
- pip
- virtualenv (recomendado)

### Docker
- Docker 20.10+
- Docker Compose 2.0+ (opcional)

##  Instalación

### Opción 1: Instalación Local

```bash
# Clonar el repositorio
git clone <repository-url>
cd breast-cancer-mlops

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

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

# Ejecutar API
make run
```

### Opción 3: Docker

```bash
# Construir imagen
docker build -t breast-cancer-api .

# Ejecutar contenedor
docker run -d -p 5000:5000 --name bc-api breast-cancer-api
```

##  Uso

### Verificar Estado del Servicio

```bash
curl http://localhost:5000/
```

**Respuesta esperada:**
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

### Realizar Predicción

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

**Respuesta esperada:**
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

### Prueba Automatizada con Script

```bash
python test_requests.py http://localhost:5000
```

## 📡 Endpoints de la API

### GET `/`
Health check del servicio.

**Response:**
- `200 OK`: Servicio operativo
- `503 Service Unavailable`: Modelo no cargado

### POST `/predict`
Realiza una predicción.

**Request Body:**
```json
{
  "features": [float, float, ..., float]  // 30 valores numéricos
}
```

**Response:**
- `200 OK`: Predicción exitosa
- `400 Bad Request`: Datos inválidos
- `503 Service Unavailable`: Modelo no disponible

### GET `/model-info`
Obtiene información detallada del modelo.

**Response:**
```json
{
  "status": "success",
  "model_metadata": {
    "trained_at": "2024-01-15T10:30:00",
    "test_auc": 0.9876,
    "best_params": {...},
    "top_features": [...]
  }
}
```

## 🐳 Docker

### Construcción de Imagen

```bash
# Construir imagen
docker build -t breast-cancer-api:latest .

# Verificar imagen
docker images | grep breast-cancer-api
```

### Ejecución del Contenedor

```bash
# Ejecutar contenedor
docker run -d \
  -p 5000:5000 \
  --name breast-cancer-api \
  --restart unless-stopped \
  breast-cancer-api:latest

# Ver logs
docker logs -f breast-cancer-api

# Detener contenedor
docker stop breast-cancer-api

# Eliminar contenedor
docker rm breast-cancer-api
```

### Docker Compose

```bash
# Iniciar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

##  CI/CD Pipeline

El proyecto incluye un pipeline completo de CI/CD con GitHub Actions que:

1. **Test Job**
   - Instala dependencias
   - Verifica estructura del proyecto
   - Entrena el modelo (si no existe)
   - Ejecuta tests unitarios

2. **Build Job**
   - Construye la imagen Docker
   - Ejecuta tests del contenedor
   - Guarda la imagen como artefacto

3. **Push Job** (Opcional)
   - Pushea la imagen a Docker Hub
   - Tagea con SHA del commit

### Configurar CI/CD

1. Fork o clona el repositorio
2. Habilita GitHub Actions
3. (Opcional) Configura secretos para Docker Hub:
   - `DOCKER_USERNAME`
   - `DOCKER_PASSWORD`

##  Testing

### Ejecutar Tests Unitarios

```bash
# Con pytest
pytest tests/ -v

# Con coverage
pytest tests/ -v --cov=. --cov-report=html

# Con make
make test
```

### Tests Incluidos

- ✅ Health check endpoint
- ✅ Predicción con datos válidos
- ✅ Validación de entradas
- ✅ Manejo de errores
- ✅ Tests de endpoints adicionales

## 🤖 Modelo de Machine Learning

### Detalles del Modelo

- **Algoritmo**: Random Forest Classifier
- **Dataset**: Breast Cancer Wisconsin (569 muestras, 30 características)
- **Métricas de Performance**:
  - AUC-ROC: ~0.98
  - Precisión: ~96%
  - Recall: ~95%

### Características del Dataset

Las 30 características incluyen mediciones de:
- Radio
- Textura
- Perímetro
- Área
- Suavidad
- Compacidad
- Concavidad
- Puntos cóncavos
- Simetría
- Dimensión fractal

Cada característica tiene: media, error estándar y "worst" (promedio de los 3 valores más grandes).

### Preprocesamiento

1. **Normalización**: StandardScaler para todas las características
2. **División**: 80% entrenamiento, 20% prueba
3. **Validación Cruzada**: 5-fold CV para optimización

### Optimización de Hiperparámetros

Grid Search sobre:
- `n_estimators`: [100, 200]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5]
- `min_samples_leaf`: [1, 2]

## 📁 Estructura del Proyecto

```
breast-cancer-mlops/

├── model/
│   ├── breast_cancer_model.pkl    # Modelo entrenado
│   └── model_metadata.json        # Metadata del modelo
│
├── tests/
│   └── test_api.py            # Tests unitarios
│
├── app.py                     # API Flask
├── train_model.py             # Script de entrenamiento
├── test_requests.py           # Tests manuales
├── requirements.txt           # Dependencias Python
├── Dockerfile                 # Configuración Docker
├── docker-compose.yml         # Orquestación Docker
├── Makefile                   # Comandos útiles
└── README.md                  # Este archivo
```

##  Troubleshooting

### Error: Modelo no encontrado

```bash
# Solución: Entrenar el modelo
python train_model.py
```

### Error: Puerto 5000 ocupado

```bash
# Cambiar puerto en app.py o Docker
docker run -p 8000:5000 breast-cancer-api
```

### Error: Tests fallando

```bash
# Asegurarse de tener el modelo entrenado
python train_model.py

# Reinstalar dependencias
pip install -r requirements.txt
```

##  Buenas Prácticas Implementadas

-  Separación de concerns (training/serving)
-  Validación de entradas
-  Manejo de errores robusto
-  Logging estructurado
-  Tests automatizados
-  Documentación completa
-  Versionado de código
-  Containerización
-  CI/CD automatizado
-  Health checks
-  Seguridad (usuario no-root en Docker)


