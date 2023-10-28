import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import convolve, lfilter, firwin, medfilt, savgol_filter, wiener
from scipy.linalg import toeplitz

# instalar librerias con pip install
# pip install numpy
# pip install pandas
# pip install matplotlib
# pip install scipy


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
    graficar(t, x, y)  # llama a la función graficar


# crear funcion para graficar las ondas
def graficar(t, x, y):
    global df_uniforme, tiempo_uniforme, onda_reconstruida_y  # declarar las variables como globales
    # t, x, y = cargar_archivo()
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
    # Definir el tamaño del padding
    padding_size = window_size // 2

    # Crear los datos de padding
    padding_x = np.pad(x_uniforme, (padding_size, padding_size), "edge")
    padding_y = np.pad(y_uniforme, (padding_size, padding_size), "edge")

    # Aplicar el filtro de media móvil a la señal con padding
    x_filtered = convolve(padding_x, window, mode="same")
    y_filtered = convolve(padding_y, window, mode='same')
    #y_filtered = medfilt(y_uniforme, kernel_size=5)
    # Aplicar el filtro Wiener a la señal
    #x_filtered = wiener(x_uniforme)
    #y_filtered = wiener(y_uniforme)

    '''
    x_filtered = savgol_filter(x_uniforme, window_length=5, polyorder=2)
    y_filtered = savgol_filter(y_uniforme, window_length=5, polyorder=2)
    '''
    # Eliminar el padding de la señal filtrada
    x_filtered = x_filtered[padding_size:-padding_size]
    y_filtered = y_filtered[padding_size:-padding_size]
    '''
    min_length = min(len(tiempo_uniforme), len(x_filtered), len(y_filtered))
    tiempo_uniforme = tiempo_uniforme[:min_length]
    x_filtered = x_filtered[:min_length]
    y_filtered = y_filtered[:min_length]
    '''
    # Identificar las frecuencias dominantes
    indices_dominantes_x = np.argsort(np.abs(fftx))[-30:]
    indices_dominantes_y = np.argsort(np.abs(ffty))[-20:]

    # Inicializar la onda reconstruida como un array de ceros
    onda_reconstruida_x = np.zeros_like(tiempo)
    onda_reconstruida_y = np.zeros_like(tiempo)
    # Graficar la onda original x
    plt.figure(figsize=(20, 15))
    plt.subplot(4, 3, 1)
    plt.plot(tiempo, x, label="Original X")
    # Graficar la onda filtrada
    plt.plot(tiempo_uniforme, x_filtered, label="Interpolada")
    plt.title("Onda filtrada X")
    plt.legend()
    # Graficar la onda original y
    plt.subplot(4, 3, 2)
    plt.plot(tiempo, y, label="Original Y")
    # Graficar la onda filtrada
    plt.plot(tiempo_uniforme, y_filtered, label="Interpolada")
    plt.title("Onda filtrada Y")
    plt.legend()
    # Graficar el espectro de Fourier x e y
    plt.subplot(4, 3, 3)
    plt.plot(freqx, np.abs(fftx))
    plt.title("Espectro de Fourier X")
    plt.subplot(4, 3, 4)
    plt.plot(freqy, np.abs(ffty))
    plt.title("Espectro de Fourier Y")
    plt.subplot(4, 3, 5)
    # Graficar cada componente sinusoidal individualmente en x
    for i in zip(indices_dominantes_x):
        # Calcular la amplitud y la fase de la componente sinusoidal
        amplitud = np.abs(fftx[i]) / len(fftx)
        fase = -np.angle(fftx[i])
        # Calcular la onda sinusoidal individual
        onda_individual = amplitud * np.cos(
            2 * np.pi * freqx[i] * tiempo - fase
        )
        # Sumar la onda individual a la onda reconstruida
        onda_reconstruida_x += onda_individual
        plt.plot(tiempo, onda_individual)
    plt.title("Ondas Individuales X")
    # Graficar cada componente sinusoidal individualmente en y
    plt.subplot(4, 3, 6)
    for i in zip(indices_dominantes_y):
        # Calcular la amplitud y la fase de la componente sinusoidal
        amplitud = np.abs(ffty[i]) / len(ffty)
        fase = -np.angle(ffty[i])
        # Calcular la onda sinusoidal individual
        onda_individual = amplitud * np.cos(
            2 * np.pi * freqy[i] * tiempo - fase
        )
        # Sumar la onda individual a la onda reconstruida
        onda_reconstruida_y += onda_individual
        plt.plot(tiempo, onda_individual)
    plt.title("Ondas Individuales Y")
    # Calcular la amplitud máxima de la señal original
    amplitud_max_original_y = np.max(np.abs(y))
    amplitud_max_original_x = np.max(np.abs(x))

    # Calcular la amplitud máxima de la señal reconstruida
    amplitud_max_reconstruida_y = np.max(np.abs(onda_reconstruida_y))
    amplitud_max_reconstruida_x = np.max(np.abs(onda_reconstruida_x))

    # Normalizar la señal reconstruida para que tenga la misma amplitud que la señal original
    onda_reconstruida_normalizada_y = onda_reconstruida_y * (amplitud_max_original_y / amplitud_max_reconstruida_y)
    onda_reconstruida_normalizada_x = onda_reconstruida_x * (amplitud_max_original_x / amplitud_max_reconstruida_x)

    # Crear un DataFrame con tus datos
    df_uniforme = pd.DataFrame({"t": tiempo, "x": onda_reconstruida_x, "y": onda_reconstruida_y})

    plt.subplot(4, 3, 7)
    plt.title("Onda reconstruida Y")
    plt.plot(tiempo, onda_reconstruida_y, label="Reconstruida Y")
    plt.plot(tiempo, y, label="Original Y")
    plt.legend()
    plt.subplot(4, 3, 8)
    plt.title("Onda reconstruida X")
    plt.plot(tiempo, onda_reconstruida_x, label="Reconstruida X")
    plt.plot(tiempo, x, label="Original X")
    plt.legend()
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
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
    root, text="Cargar archivo", command=cargar_archivo
)  # crea un botón que llama a la función cargar_archivo cuando se presiona
# crear boton para graficar
boton2 = tk.Button(root, text="Graficar", command=graficar)
# crea boton para guardar archivo
boton3 = tk.Button(root, text="Guardar archivo", command=guardar_archivo)
boton.pack()
boton2.pack()
boton3.pack()

root.mainloop()
