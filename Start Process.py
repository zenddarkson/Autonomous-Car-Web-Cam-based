import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import PIL
from PIL import Image
import torch.optim.lr_scheduler as lr_scheduler
import os
import random
import cv2
import time
import serial

#############Función que define el modelo convolucional

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # #entrada, # salida, el tamaño del kernel, el paso y el relleno.
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)

        # Calcular el tamaño de entrada para la capa de regresión
        self.fc_input_size = 64 * 1 * 1  # Ajustado para un tamaño de salida válido

        # Capa totalmente conectada para regresión
        self.fc1 = nn.Linear(self.fc_input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Aplicar convoluciones y funciones de activación ReLU
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)  # Max pooling para reducir el tamaño espacial
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)  # Max pooling para reducir el tamaño espacial
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2)  # Max pooling para reducir el tamaño espacial
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)  # Max pooling para reducir el tamaño espacial
        x = F.relu(self.conv5(x))

        # Aplanar la salida para la capa de regresión
        x = x.view(-1, self.fc_input_size)

        # Capas totalmente conectadas para regresión
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # Sin función de activación en la última capa para regresión

        return x

# Crear una instancia del modelo
model = ConvNet()

import time
# Función para cargar el modelo
def load_model(model_path):
    model = ConvNet()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def preprocess(img):
    grayscale = transforms.Grayscale()
    img = grayscale(img)
    preprocess = transforms.Compose([
        transforms.Resize((220, 220)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)
    return img_tensor

# Función para realizar la predicción
def predict(model, img_tensor):
    with torch.no_grad():
        output = model(img_tensor)
        steering_angle = output.item()
    return steering_angle

def suavizar_imagen(image_pil):

    # Convertir la imagen PIL a un array NumPy
    image_np = np.array(image_pil)

    # Aplicar el suavizado utilizando el filtro Gaussiano de OpenCV
    smooth_image_np = cv2.GaussianBlur(image_np, (5, 5), 0)

    # Convertir el array NumPy de vuelta a un objeto PIL
    smooth_image_pil = Image.fromarray(smooth_image_np)

    return smooth_image_pil


#################################################################################################



# Cargar el modelo
model = load_model('C:/Users/elmer/Downloads/DataTraining2/modelos/model_1.pth')

# Preprocesar la imagen y realizar la predicción
start_time = time.time()
ubicacion = 'C:/Users/elmer/Downloads/DataTraining2/TESTEO/3.jpg'
image_pil = Image.open(ubicacion)
smoothed_frame = suavizar_imagen(image_pil)
img_tensor = preprocess(smoothed_frame)
steering_angle = predict(model, img_tensor)
end_time = time.time()
prediction_time = end_time - start_time

# Mostrar la imagen y la predicción
plt.imshow(Image.open(ubicacion))
plt.title(f'Dirección: {steering_angle}')
plt.show()
print(prediction_time)
print(type(steering_angle))  # Imprimirá <class 'int'>


#####################################################################################################

# Inicialización del puerto serial
#ser = serial.Serial('/dev/ttyTHS1', 9600)
ser = serial.Serial('COM5', 9600)
time.sleep(2)  # Espera inicial para establecer la conexión serial

def capture_frames(model):
    camera = cv2.VideoCapture(0)
    interval = 0.1  # Intervalo objetivo entre capturas, en segundos
    last_speed = 100  # Valor inicial de velocidad
    last_angle = 0  # Valor inicial de ángulo

    try:
        while True:
            start_time = time.time()

            success, frame = camera.read()
            if success:
                # Preprocesar la imagen (si es necesario)
                resized = preprocess(frame)
                # Pasar la imagen al modelo de IA para obtener un valor
                angle = pytorch_predict(resized)
                angle = angle * 2  # Factores de adaptación
                angle = int(angle)  # Convertir a entero
                speed = 100  # Valor de velocidad (puede modificarse según sea necesario)

                last_speed = speed  # Actualizar el último valor de velocidad
                last_angle = angle  # Actualizar el último valor de ángulo
            else:
                # Si falla la captura, enviar el último valor de velocidad y ángulo conocidos
                speed = last_speed
                angle = last_angle
                print("Fallo al capturar el fotograma. Enviando último valor conocido...")

            # Enviar los datos al puerto serial
            ser.write(f"{speed},{angle}\n".encode())
            print(f"Enviando datos - Velocidad: {speed}, Ángulo: {angle}")
            if success:
                # Mostrar el fotograma en una ventana solo si la captura fue exitosa
                cv2.imshow('Video', frame)

            end_time = time.time()
            duration = end_time - start_time

            # Esperar el tiempo restante para cumplir con el intervalo, si es necesario
            if duration < interval:
                time.sleep(interval - duration)

            # Presionar 'q' para salir del bucle
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        # Enviar un último valor de velocidad y ángulo al detener la ejecución
        ser.write("0,0\n".encode())  # Indicar detención
        print("Deteniendo sistema. Enviando velocidad 0, ángulo 0.")
        # Limpieza
        camera.release()
        cv2.destroyAllWindows()
        ser.close()  # Cerrar el puerto serial

# Capturar y mostrar los fotogramas
capture_frames(model=None)  # Puedes pasar un modelo de PyTorch si lo deseas