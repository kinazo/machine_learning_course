"""
Script de entrenamiento del modelo de clasificación de cáncer de mama.
Utiliza el dataset Breast Cancer Wisconsin y entrena un modelo Random Forest.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import json
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BreastCancerModel:
    """Clase para entrenar y gestionar el modelo de clasificación."""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.metadata = {}
        
    def load_data(self):
        """Carga y prepara el dataset de cáncer de mama."""
        logger.info("Cargando dataset Breast Cancer Wisconsin...")
        data = load_breast_cancer()
        
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name='target')
        
        self.feature_names = list(data.feature_names)
        
        logger.info(f"Dataset cargado: {X.shape[0]} muestras, {X.shape[1]} características")
        logger.info(f"Distribución de clases: {dict(y.value_counts())}")
        
        return X, y
    
    def preprocess_data(self, X_train, X_test):
        """Normaliza las características."""
        logger.info("Normalizando características...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def train(self, X_train, y_train):
        """Entrena el modelo con optimización de hiperparámetros."""
        logger.info("Iniciando entrenamiento del modelo...")
        
        # Definir parámetros para GridSearch
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Modelo base
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Búsqueda de mejores hiperparámetros
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='roc_auc', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.metadata['best_params'] = grid_search.best_params_
        self.metadata['best_score'] = float(grid_search.best_score_)
        
        logger.info(f"Mejores parámetros: {grid_search.best_params_}")
        logger.info(f"Mejor score CV: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo en el conjunto de prueba."""
        logger.info("Evaluando modelo...")
        
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Métricas
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        logger.info(f"AUC-ROC Score: {auc_score:.4f}")
        logger.info("\nReporte de clasificación:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Guardar métricas en metadata
        self.metadata['test_auc'] = float(auc_score)
        self.metadata['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
        
        # Importancia de características (top 10)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        self.metadata['top_features'] = feature_importance.to_dict('records')
        
        logger.info("\nTop 10 características más importantes:")
        logger.info(f"\n{feature_importance}")
        
        return auc_score
    
    def save_model(self, model_path='model/breast_cancer_model.pkl'):
        """Guarda el modelo, scaler y metadata."""
        logger.info(f"Guardando modelo en {model_path}...")
        
        # Agregar timestamp
        self.metadata['trained_at'] = datetime.now().isoformat()
        self.metadata['feature_names'] = self.feature_names
        
        # Guardar modelo y scaler
        model_artifacts = {
            'model': self.model,
            'scaler': self.scaler,
            'metadata': self.metadata
        }
        
        joblib.dump(model_artifacts, model_path)
        
        # Guardar metadata en JSON para fácil lectura
        with open('model/model_metadata.json', 'w') as f:
            json.dump(self.metadata, f, indent=4)
        
        logger.info("Modelo guardado exitosamente!")


def main():
    """Función principal de entrenamiento."""
    
    # Crear directorio de modelos
    import os
    os.makedirs('model', exist_ok=True)
    
    # Inicializar modelo
    bc_model = BreastCancerModel()
    
    # Cargar datos
    X, y = bc_model.load_data()
    
    # Dividir en train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocesar
    X_train_scaled, X_test_scaled = bc_model.preprocess_data(X_train, X_test)
    
    # Entrenar
    bc_model.train(X_train_scaled, y_train)
    
    # Evaluar
    bc_model.evaluate(X_test_scaled, y_test)
    
    # Guardar
    bc_model.save_model()
    
    logger.info("\n" + "="*50)
    logger.info("Entrenamiento completado exitosamente!")
    logger.info("="*50)


if __name__ == "__main__":
    main()
