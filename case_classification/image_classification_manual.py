import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import shap
import numpy as np
import matplotlib.pyplot as plt
import os
import json

class MNISTClassifier:
    def __init__(self, config):
        self.model = None
        self.history = None
        self.config = config
        self.x_train, self.y_train, self.x_test, self.y_test = self.load_data()
        self.input_shape = self.x_train.shape[1:]

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

        # Configurar el optimizador
        optimizer = self.get_optimizer()

        self.model.compile(optimizer=optimizer,
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def get_optimizer(self):
        opt_name = self.config.get("optimizer", "adam")
        learning_rate = self.config.get("learning_rate", 0.001)
        if opt_name.lower() == "adam":
            return optimizers.Adam(learning_rate=learning_rate)
        elif opt_name.lower() == "sgd":
            return optimizers.SGD(learning_rate=learning_rate)
        elif opt_name.lower() == "rmsprop":
            return optimizers.RMSprop(learning_rate=learning_rate)
        else:
            raise ValueError(f"Optimizer '{opt_name}' not recognized")

    def get_callbacks(self):
        callbacks = []

        # Registrar métricas por época
        class LogMetricsCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs:
                    for key, value in logs.items():
                        mlflow.log_metric(key, value, step=epoch)

        callbacks.append(LogMetricsCallback())

        # Configurar el scheduler si está especificado
        scheduler_config = self.config.get("scheduler", None)
        if scheduler_config:
            scheduler_type = scheduler_config.get("type")
            if scheduler_type == "exponential":
                decay_rate = scheduler_config.get("decay_rate", 0.9)
                def scheduler(epoch, lr):
                    return lr * decay_rate
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
            elif scheduler_type == "step":
                step_size = scheduler_config.get("step_size", 1)
                gamma = scheduler_config.get("gamma", 0.1)
                def scheduler(epoch, lr):
                    if epoch % step_size == 0 and epoch:
                        return lr * gamma
                    return lr
                callbacks.append(tf.keras.callbacks.LearningRateScheduler(scheduler))
            else:
                raise ValueError(f"Scheduler type '{scheduler_type}' not recognized")

        return callbacks

    def train(self):
        epochs = self.config.get("epochs", 5)
        batch_size = self.config.get("batch_size", 128)

        # Registrar parámetros
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("optimizer", self.config.get("optimizer", "adam"))
        mlflow.log_param("learning_rate", self.config.get("learning_rate", 0.001))
        mlflow.log_param("scheduler", self.config.get("scheduler", None))

        # Obtener callbacks
        callbacks = self.get_callbacks()

        # Entrenar el modelo
        self.history = self.model.fit(self.x_train, self.y_train,
                                      validation_data=(self.x_test, self.y_test),
                                      epochs=epochs,
                                      batch_size=batch_size,
                                      callbacks=callbacks)

        # Guardar el modelo como un modelo de Keras en MLflow
        mlflow.keras.log_model(self.model, artifact_path="model")

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
        script_dir = os.path.dirname(os.path.abspath(__file__))
        shap_images_dir = os.path.join(script_dir, "shap_images")
        os.makedirs(shap_images_dir, exist_ok=True)
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
            shap_image = os.path.join(shap_images_dir, f"shap_explanation_{i}.png")
            # Obtener los shap_values de la clase predicha
            shap_value = shap_values[predicted_classes[i]][i]
            # Usar shap.image_plot para visualizar la explicación
            shap.image_plot([shap_value], -test_images[i], show=False)
            plt.savefig(shap_image, bbox_inches='tight')
            plt.close()
            # Registrar la imagen en MLflow
            mlflow.log_artifact(shap_image, artifact_path="shap_images")

    def run(self):
        # Iniciar una nueva ejecución en MLflow
        with mlflow.start_run():
            self.build_model()
            self.train()
            self.evaluate()
            self.explain()

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
    mlflow.set_experiment("MNIST_Classification_Experiments")

    # Ejecutar un experimento por cada configuración
    for idx, config in enumerate(configurations):
        print(f"Ejecutando configuración {idx + 1}/{len(configurations)}: {config}")
        classifier = MNISTClassifier(config)
        classifier.run()

if __name__ == "__main__":
    main()


