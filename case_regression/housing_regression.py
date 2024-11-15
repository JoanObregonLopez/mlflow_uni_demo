import os
import json
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import shap
import matplotlib.pyplot as plt

class HousingRegressor:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.pipeline = None
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.feature_engineering_steps = []

    def load_data(self):
        # Cargar el dataset de California Housing
        data = fetch_california_housing(as_frame=True)
        df = data.frame
        return df

    def preprocess_data(self, df):
        # Dividir en características y variable objetivo
        X = df.drop('MedHouseVal', axis=1)
        y = df['MedHouseVal']

        # Dividir en conjuntos de entrenamiento y prueba
        test_size = self.config.get('test_size', 0.2)
        random_state = self.config.get('random_state', 42)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

    def build_pipeline(self):
        steps = []
        features = self.config.get('features', {})
        numeric_features = features.get('numeric', list(self.X_train.select_dtypes(include=[np.number]).columns))
        categorical_features = features.get('categorical', list(self.X_train.select_dtypes(include=['object']).columns))

        # Transformaciones numéricas
        numeric_transformers = []
        if self.config.get('feature_engineering', {}).get('scaling', False):
            numeric_transformers.append(('scaler', StandardScaler()))
            self.feature_engineering_steps.append('Scaling')

        if self.config.get('feature_engineering', {}).get('polynomial_features', False):
            degree = self.config['feature_engineering'].get('polynomial_degree', 2)
            numeric_transformers.append(('poly', PolynomialFeatures(degree=degree, include_bias=False)))
            self.feature_engineering_steps.append(f'PolynomialFeatures(degree={degree})')

        if numeric_transformers:
            numeric_pipeline = Pipeline(steps=numeric_transformers)
        else:
            numeric_pipeline = 'passthrough'

        # Transformaciones categóricas
        categorical_transformers = []
        if categorical_features:
            categorical_transformers.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
            self.feature_engineering_steps.append('OneHotEncoding')

        if categorical_transformers:
            categorical_pipeline = Pipeline(steps=categorical_transformers)
        else:
            categorical_pipeline = 'passthrough'

        # Componer transformaciones
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_pipeline, numeric_features),
                ('cat', categorical_pipeline, categorical_features)
            ]
        )

        # Modelo
        model = self.get_model()

        # Pipeline completo
        self.pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])

    def get_model(self):
        model_config = self.config.get('model', {})
        model_type = model_config.get('type', 'linear_regression')

        if model_type == 'linear_regression':
            self.feature_engineering_steps.append('LinearRegression')
            return LinearRegression()
        elif model_type == 'random_forest':
            n_estimators = model_config.get('n_estimators', 100)
            max_depth = model_config.get('max_depth', None)
            min_samples_split = model_config.get('min_samples_split', 2)
            self.feature_engineering_steps.append(f'RandomForestRegressor(n_estimators={n_estimators}, max_depth={max_depth})')
            return RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=self.config.get('random_state', 42)
            )
        else:
            raise ValueError(f"Modelo '{model_type}' no soportado")

    def train(self):
        # Registrar parámetros
        mlflow.log_param('model_type', self.config.get('model', {}).get('type', 'linear_regression'))
        mlflow.log_param('model_params', self.config.get('model', {}))
        mlflow.log_param('feature_engineering_steps', self.feature_engineering_steps)
        mlflow.log_param('test_size', self.config.get('test_size', 0.2))
        mlflow.log_param('random_state', self.config.get('random_state', 42))

        # Entrenar el modelo
        self.pipeline.fit(self.X_train, self.y_train)

        # Guardar el modelo en MLflow
        mlflow.sklearn.log_model(self.pipeline, artifact_path='model')

    def evaluate(self):
        # Predecir en el conjunto de prueba
        y_pred = self.pipeline.predict(self.X_test)

        # Calcular métricas
        mse = mean_squared_error(self.y_test, y_pred)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)

        # Registrar métricas
        mlflow.log_metric('mse', mse)
        mlflow.log_metric('mae', mae)
        mlflow.log_metric('r2_score', r2)

        print(f'MSE: {mse}, MAE: {mae}, R2 Score: {r2}')

    def log_feature_importance(self):
        # Obtener el modelo entrenado
        model = self.pipeline.named_steps['regressor']

        # Obtener los nombres de las características después del preprocesamiento
        feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()

        # Inicializar variables
        importances = None

        # Comprobar si el modelo tiene el atributo feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            print("El modelo no tiene atributos 'feature_importances_' ni 'coef_'")
            return

        # Crear un DataFrame con las importancias
        feat_imp_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)

        # Guardar el DataFrame como CSV
        feat_imp_csv = 'feature_importance.csv'
        feat_imp_df.to_csv(feat_imp_csv, index=False)
        mlflow.log_artifact(feat_imp_csv)

        # Crear una gráfica de barras
        plt.figure(figsize=(10, 6))
        plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
        plt.gca().invert_yaxis()
        plt.xlabel('Importance')
        plt.title('Feature Importances')
        plt.tight_layout()

        # Guardar la gráfica
        feat_imp_png = 'feature_importance.png'
        plt.savefig(feat_imp_png)
        plt.close()
        mlflow.log_artifact(feat_imp_png)

    def log_shap_values(self):
        # Crear un directorio para las imágenes si no existe
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shap_images_dir = os.path.join(script_dir, "shap_images")
        os.makedirs(shap_images_dir, exist_ok=True)

        # Utilizar un subconjunto de los datos de prueba para acelerar el cálculo
        X_test_sample = self.X_test.sample(n=100, random_state=self.config.get('random_state', 42))
        y_test_sample = self.y_test.loc[X_test_sample.index]

        # Obtener los datos transformados
        X_transformed = self.pipeline.named_steps['preprocessor'].transform(X_test_sample)
        feature_names = self.pipeline.named_steps['preprocessor'].get_feature_names_out()
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names)

        # Obtener el modelo entrenado
        model = self.pipeline.named_steps['regressor']

        # Crear el explainer de SHAP
        # Usar el explainer apropiado según el tipo de modelo
        if self.config.get('model', {}).get('type', 'linear_regression') == 'random_forest':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, X_transformed)

        # Calcular los valores SHAP
        shap_values = explainer.shap_values(X_transformed)

        # Crear un resumen de los valores SHAP
        shap_summary_png = os.path.join(shap_images_dir, 'shap_summary_plot.png')
        shap.summary_plot(shap_values, features=X_transformed_df, feature_names=feature_names, show=False)
        plt.savefig(shap_summary_png, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_summary_png, artifact_path='shap_plots')

        # Crear un gráfico de dependencia para la característica más importante
        shap_dependence_png = os.path.join(shap_images_dir, 'shap_dependence_plot.png')
        top_feature_index = np.argmax(np.abs(shap_values).mean(0))
        feature_name = feature_names[top_feature_index]
        shap.dependence_plot(feature_name, shap_values, X_transformed_df, feature_names=feature_names, show=False)
        plt.savefig(shap_dependence_png, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact(shap_dependence_png, artifact_path='shap_plots')


    def run(self):
        # Iniciar una ejecución en MLflow
        with mlflow.start_run():
            df = self.load_data()
            self.preprocess_data(df)
            self.build_pipeline()
            self.train()
            self.evaluate()
            # Registrar las transformaciones de feature engineering
            mlflow.log_param('feature_engineering', ', '.join(self.feature_engineering_steps))
            # Registrar la importancia de características si es posible
            self.log_feature_importance()
            # Registrar los valores SHAP
            self.log_shap_values()

def main():
    # Obtener la ruta del directorio donde se encuentra el script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construir la ruta completa al archivo configurations.json
    config_path = os.path.join(script_dir, 'configurations.json')

    # Comprobar si el archivo existe
    if not os.path.exists(config_path):
        print(f"El archivo 'configurations.json' no se encontró en {script_dir}")
        return

    # Cargar las configuraciones desde el archivo JSON
    with open(config_path, 'r') as f:
        configurations = json.load(f)

    # Establecer el experimento (opcional)
    mlflow.set_experiment("Housing_Price_Prediction")

    # Ejecutar un experimento por cada configuración
    for idx, config in enumerate(configurations):
        print(f"Ejecutando configuración {idx + 1}/{len(configurations)}: {config}")
        regressor = HousingRegressor(config)
        regressor.run()

if __name__ == '__main__':
    main()

