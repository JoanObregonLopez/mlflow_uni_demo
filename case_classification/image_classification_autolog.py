import mlflow
import mlflow.tensorflow
import tensorflow as tf
from tensorflow.keras import layers, models
import shap
import numpy as np
import matplotlib.pyplot as plt

class MNISTClassifier:
    def __init__(self):
        self.model = None
        self.history = None
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
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self, epochs=5):
        # Iniciar una nueva ejecución en MLflow
        with mlflow.start_run():
            mlflow.tensorflow.autolog()
            self.history = self.model.fit(self.x_train, self.y_train,
                                          validation_data=(self.x_test, self.y_test),
                                          epochs=epochs,
                                          batch_size=128)
            # Registrar parámetros adicionales si es necesario
            mlflow.log_param("epochs", epochs)

    def evaluate(self):
        # Evaluar el modelo en el conjunto de prueba
        test_loss, test_acc = self.model.evaluate(self.x_test, self.y_test)
        print(f'Pérdida en test: {test_loss}, Precisión en test: {test_acc}')

    def explain(self):
        # Usar SHAP para explicar el modelo
        # Seleccionar un subconjunto de datos para la explicación
        background = self.x_train[np.random.choice(self.x_train.shape[0], 100, replace=False)]
        test_images = self.x_test[:5]

        # Crear el explainer
        explainer = shap.DeepExplainer(self.model, background)
        shap_values = explainer.shap_values(test_images)

        # Graficar las explicaciones para cada clase
        for i in range(len(test_images)):
            plt.figure(figsize=(10, 3))
            for j in range(10):
                plt.subplot(2, 5, j+1)
                shap.image_plot([shap_values[j][i]], -test_images[i])
            plt.show()

if __name__ == "__main__":
    classifier = MNISTClassifier()
    classifier.build_model()
    classifier.train(epochs=5)
    classifier.evaluate()
    classifier.explain()
