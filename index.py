import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, lfilter

def cargar_archivo():
    ruta_archivo = (
        filedialog.askopenfilename()
    )  # abre el explorador de archivos y guarda la ruta del archivo seleccionado
    if ruta_archivo:  # si se seleccionó un archivo
        df = pd.read_excel(ruta_archivo)  # lee el archivo con pandas
        print(df)  # imprime el contenido del archivo
        t = df["t"]
        x = df["x"]
        y = df["y"]
    graficar(t,x,y)  # llama a la función graficar




# crear funcion para graficar las ondas
def graficar(t,x,y):
    global df_uniforme, tiempo_uniforme, onda_reconstruida_y  # declarar las variables como globales
    #t, x, y = cargar_archivo()
    tiempo = np.array(t)
    # Convertir la lista tiempo a un objeto de tipo numpy.ndarray
    # Crear un nuevo array de tiempo con espaciado uniforme
    tiempo_uniforme = np.linspace(tiempo.min(), tiempo.max(), len(tiempo))
    # Interpolar los datos de la onda para que estén uniformemente espaciados en x e y
    x_uniforme = np.interp(tiempo_uniforme, tiempo, x)
    y_uniforme = np.interp(tiempo_uniforme, tiempo, y)
    # FFT de x e y
    fftx = np.fft.fft(x_uniforme)
    ffty = np.fft.fft(y_uniforme)
    # Frecuencia de x e y
    freqx = np.fft.fftfreq(len(x), tiempo_uniforme[1] - tiempo_uniforme[0])
    freqy = np.fft.fftfreq(len(y), tiempo_uniforme[1] - tiempo_uniforme[0])
    # Definir el tamaño de la ventana del filtro
    window_size = 10
    # Crear la ventana del filtro
    window = np.ones(window_size) / window_size
    # Aplicar el filtro de media móvil a la señal
    x_filtered = convolve(x, window, mode='same')
    y_filtered = convolve(y, window, mode='same')
    # Crear un DataFrame con tus datos
    df_uniforme = pd.DataFrame({"t": tiempo_uniforme, "x": x_filtered, "y": y_filtered})
    # Inicializar la onda reconstruida como un array de ceros
    onda_reconstruida_x = np.zeros_like(tiempo_uniforme)
    onda_reconstruida_y = np.zeros_like(tiempo_uniforme)
    # Graficar la onda original x
    plt.figure(figsize=(20, 12))
    plt.subplot(3, 3, 1)
    plt.plot(tiempo, x, label="Original X")
    # Graficar la onda filtrada
    plt.plot(tiempo_uniforme, x_filtered, label="Interpolada")
    plt.title("Onda filtrada X")
    plt.legend()
    # Graficar la onda original y
    plt.subplot(3, 3, 2)
    plt.plot(tiempo, y, label="Original Y")
    # Graficar la onda filtrada
    plt.plot(tiempo_uniforme, y_filtered, label="Interpolada")
    plt.title("Onda filtrada Y")
    plt.legend()
    # Graficar el espectro de Fourier x e y
    plt.subplot(3, 3, 3)
    plt.plot(freqx, np.abs(fftx))
    plt.title("Espectro de Fourier X")
    plt.subplot(3, 3, 4)
    plt.plot(freqy, np.abs(ffty))
    plt.title("Espectro de Fourier Y")
    plt.subplot(3, 3, 5)
    # Graficar cada componente sinusoidal individualmente en x
    for i in range(len(fftx)):
        # Calcular la amplitud y la fase de la componente sinusoidal
        amplitud = np.abs(fftx[i]) / len(fftx)
        fase = -np.angle(fftx[i])
        # Calcular la onda sinusoidal individual
        onda_individual = amplitud * np.cos(
            2 * np.pi * freqx[i] * tiempo_uniforme - fase
        )
        # Sumar la onda individual a la onda reconstruida
        onda_reconstruida_x += onda_individual
        plt.plot(tiempo_uniforme, onda_individual)
    plt.title("Ondas Individuales X")
    # Graficar cada componente sinusoidal individualmente en y
    plt.subplot(3, 3, 6)
    for i in range(len(ffty)):
        # Calcular la amplitud y la fase de la componente sinusoidal
        amplitud = np.abs(ffty[i]) / len(ffty)
        fase = -np.angle(ffty[i])
        # Calcular la onda sinusoidal individual
        onda_individual = amplitud * np.cos(
            2 * np.pi * freqy[i] * tiempo_uniforme - fase
        )
        # Sumar la onda individual a la onda reconstruida
        onda_reconstruida_y += onda_individual
        plt.plot(tiempo_uniforme, onda_individual)
    plt.title("Ondas Individuales Y")
    plt.show()

# funcion para solicitar nombre de archivo
def guardar_archivo():
    ruta_archivo = (
        filedialog.asksaveasfilename()
    )  # abre el explorador de archivos y guarda la ruta del archivo seleccionado
    if ruta_archivo:  # si se seleccionó un archivo
        df_uniforme.to_excel(ruta_archivo, index=False)  # lee el archivo con pandas


root = tk.Tk()
boton = tk.Button(
    root, text="Cargar archivo", command=cargar_archivo)  # crea un botón que llama a la función cargar_archivo cuando se presiona
# crear boton para graficar
boton2 = tk.Button(root, text="Graficar", command=graficar)
#crea boton para guardar archivo
boton3 = tk.Button(root, text="Guardar archivo", command=guardar_archivo)
boton.pack()
boton2.pack()
boton3.pack()

root.mainloop()