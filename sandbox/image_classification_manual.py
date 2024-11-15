import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
import shap
import numpy as np
import matplotlib.pyplot as plt
import os

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.input_shape = self.x_train.shape[1:]
        self.mlflow_run = None  # Para mantener referencia a la ejecución de MLflow

    def load_data(self):
        # Cargar el conjunto de datos MNIST
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        # Normalizar los datos
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        # Expandir dimensiones para canales (grises)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        return x_train, y_train, x_test, y_test

    def build_model(self):
        # Construir el modelo de deep learning
        self.model = models.Sequential([
            layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(10, activation='softmax')
        ])
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=5):
        # Registrar parámetros
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", 128)
        # Definir un callback para registrar métricas por época
        class LogMetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    for key, value in logs.items():
                        mlflow.log_metric(key, value, step=epoch)
        # Entrenar el modelo con el callback
        self.history = self.model.fit(self.x_train, self.y_train,
                                      validation_data=(self.x_test, self.y_test),
                                      epochs=epochs,
                                      batch_size=128,
                                      callbacks=[LogMetricsCallback()])
        # Guardar el modelo manualmente
        model_path = "model"
        self.model.save(model_path)
        # Registrar el modelo como artefacto en MLflow
        mlflow.log_artifacts(model_path, artifact_path="model")

    def evaluate(self):
        # Evaluar el modelo en el conjunto de prueba
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f'Pérdida en test: {test_loss}, Precisión en test: {test_acc}')
        # Registrar métricas finales
        mlflow.log_metric("test_loss", test_loss)
        mlflow.log_metric("test_accuracy", test_acc)
        return test_loss, test_acc

    def explain(self):
        # Crear un directorio para las imágenes si no existe
        os.makedirs("shap_images", exist_ok=True)
        # Usar SHAP para explicar el modelo
        background = self.x_train[np.random.choice(self.x_train.shape[0], 100, replace=False)]
        test_images = self.x_test[:5]
        # Crear el explainer
        explainer = shap.GradientExplainer(self.model, background)
        shap_values = explainer.shap_values(test_images)
        # Obtener las predicciones del modelo
        predictions = self.model.predict(test_images)
        predicted_classes = np.argmax(predictions, axis=1)
        # Graficar y guardar las explicaciones para cada imagen
        for i in range(len(test_images)):
            shap_image = f"shap_images/shap_explanation_{i}.png"
            # Obtener los shap_values de la clase predicha
            shap_value = shap_values[predicted_classes[i]][i]
            # Usar shap.image_plot para visualizar la explicación
            shap.image_plot([shap_value], -test_images[i], show=False)
            plt.savefig(shap_image, bbox_inches='tight')
            plt.close()
            # Registrar la imagen en MLflow
            mlflow.log_artifact(shap_image, artifact_path="shap_images")

    def run(self, epochs=5):
        # Iniciar una nueva ejecución en MLflow en el nivel superior
        with mlflow.start_run() as run:
            self.mlflow_run = run  # Guardar referencia a la ejecución
            self.build_model()
            self.train(epochs=epochs)
            self.evaluate()
            self.explain()

if __name__ == "__main__":
    classifier = MNISTClassifier()
    classifier.run(epochs=5)


