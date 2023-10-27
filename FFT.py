import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Asegúrate de tener la ruta correcta al archivo
df = pd.read_excel("onda_normalizada.xlsx")

# Supongamos que 'Tiempo' es tu columna de tiempo y 'x' es tu columna de datos de onda
tiempo = df["tiempo"]
onda_compleja = df["x"]



# Aplicar FFT
fft = np.fft.fft(onda_compleja)

# Obtener las frecuencias
freq = np.fft.fftfreq(len(onda_compleja), tiempo[1] - tiempo[0])

# Identificar las frecuencias dominantes
indices_dominantes = np.argsort(np.abs(fft))[-50:]

# Crear una figura para el gráfico
plt.figure(figsize=(16, 6))

# Graficar la señal original
plt.subplot(1, 4, 1)
plt.plot(tiempo, onda_compleja)
plt.title('Señal Original')

# Graficar el espectro de frecuencias
plt.subplot(1, 4, 2)
plt.plot(np.abs(freq), np.abs(fft))
plt.title('Espectro de Frecuencias')

# Graficar cada onda individualmente en el mismo gráfico
plt.subplot(1, 4, 3)
colores = plt.cm.viridis(np.linspace(0, 1, len(indices_dominantes)))
onda_reconstruida = np.zeros_like(onda_compleja)
for i, color in zip(indices_dominantes, colores):
    amplitud = np.abs(fft[i]) / len(fft)
    fase = np.angle(fft[i])
    onda_individual = amplitud * np.cos(2 * np.pi * freq[i] * tiempo + fase)
    #print(f'{amplitud} * cos(2π * {freq[i]} * x + {fase})+')
    print(f'{amplitud:.3f} * cos(2*pi * {freq[i]:.3f} * x + {fase:.3f})+')
    onda_reconstruida += onda_individual
    plt.plot(tiempo, onda_individual, color=color)

plt.title('Ondas Individuales')

# Calcular la amplitud máxima de la señal original
amplitud_max_original = np.max(np.abs(onda_compleja))

# Calcular la amplitud máxima de la señal reconstruida
amplitud_max_reconstruida = np.max(np.abs(onda_reconstruida))

# Normalizar la señal reconstruida para que tenga la misma amplitud que la señal original
onda_reconstruida_normalizada = onda_reconstruida * (amplitud_max_original / amplitud_max_reconstruida)


# Graficar la señal reconstruida
plt.subplot(1, 4, 4)
plt.plot(tiempo, onda_reconstruida_normalizada)
plt.title('Señal Reconstruida')

plt.show()
