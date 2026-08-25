# Este programa grafica los acelerogramas de cada estación y los guarda como PDF.

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ARCHIVO_ODS = "sismo_ago_10_2026.ods"
NOMBRE_COLUMNA = "Filename"
COLUMNA_FS = "Frec_muestreo_Hz"
CARPETA_NPY = "acelerogramas_npy"
CARPETA_PDF = "acelerogramas_pdf"

COMPONENTES = [
    "Vertical",
    "Horizontal Norte-Sur",
    "Horizontal Este-Oeste",
]

def gms_a_decimal(coordenada):
    """Convierte una coordenada en formato DDMM.mmm a grados decimales."""
    # Los dos últimos dígitos de la parte entera son los minutos enteros;
    # el resto corresponde a los grados.

    signo = np.sign(coordenada)
    coord = np.abs(coordenada)

    parte_entera = int(coord)
    parte_decimal = coord - parte_entera

    minutos_enteros = parte_entera % 100
    grados = parte_entera // 100
    minutos_totales = minutos_enteros + parte_decimal

    return signo * (grados + (minutos_totales / 60))


def formatear_coordenada(valor):
    """Convierte a texto un valor de coordenada en grados decimales, indicando si no está disponible."""
    return f"{gms_a_decimal(valor):.6f}°" if pd.notna(valor) else "N/D"

# Se lee la carpeta donde están los .npy
ruta_ods = Path(ARCHIVO_ODS).resolve()
carpeta_npy = ruta_ods.parent / CARPETA_NPY
df = pd.read_excel(ruta_ods, engine="odf")

# Se configura la carpeta de salida para los PDF 
carpeta_salida_pdf = ruta_ods.parent / CARPETA_PDF
carpeta_salida_pdf.mkdir(exist_ok=True)

# Se itera sobre cada acelerograma
for _, fila in df.iterrows():
    # Se lee el nombre del archivo
    nombre = str(fila[NOMBRE_COLUMNA]).strip()
    if not nombre:
        continue

    # A partir del nombre del archivo se lee el acelerograma
    fs = float(fila[COLUMNA_FS])             # frecuencia de muestreo [Hz]
    dt = 1/fs                               # paso de tiempo del registro [s]
    ruta_matriz = carpeta_npy / f"{nombre}.npy"
    ag = np.load(ruta_matriz)                # aceleración del suelo en [g]
    t_sismo = np.arange(len(ag)) * dt        # vector de tiempo [s]

    # Se calcula el Peak Ground Acceleration por componente
    pga = np.max(np.abs(ag), axis=0)         # PGA de cada componente [g]

    # Se calcula el Peak Ground Acceleration usando las tres componentes
    pga_total = np.max(np.linalg.norm(ag, axis=1))  # PGA total [g]
    print(f"{nombre:<20}: PGA total = {pga_total:.4f} g")

    # Etiquetas para la figura
    titulo = (
        f"Acelerograma registrado en \n{fila['Station']} "
        f"({formatear_coordenada(fila['Latitude_N'])}, "
        f"{formatear_coordenada(-fila['Longitude_W'])}, "  # W negativo -> longitud Este
        f"{int(fila['Altitude_m'])} m)"
        f" - PGA total = {pga_total:.4f} g"
    )
    #etiqueta_x = (
    #    f"Tiempo [s] (desde {fila['First_sample_UTC']} "
    #    f"hasta {fila['Last_sample_UTC']} UTC)"
    #)
    etiqueta_x = f"Tiempo [s]"

    # Se crea la figura con tres subplots, uno por componente
    fig, ejes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Se grafica cada componente del acelerograma
    for indice, componente in enumerate(COMPONENTES):
        ejes[indice].plot(t_sismo, ag[:, indice], linewidth=1.0)
        ejes[indice].axhline( pga[indice], color="red", linestyle="--", linewidth=0.8)
        ejes[indice].axhline(-pga[indice], color="red", linestyle="--", linewidth=0.8)
        ejes[indice].axvline(30, color="red", linestyle="--", linewidth=0.8)

        ejes[indice].set_title(f"Componente {componente} (PGA = {pga[indice]:.4f} g)")
        ejes[indice].set_ylabel(f"Aceleración [g]")
        ejes[indice].grid()
        ejes[indice].set_xlim(0, t_sismo[-1])

    # Se ajusta la figura y se guarda como PDF
    ejes[-1].set_xlabel(etiqueta_x)
    fig.suptitle(titulo)
    fig.tight_layout()
    fig.savefig(carpeta_salida_pdf / f"{nombre}.pdf")
    plt.close(fig)
    print(f"{nombre:<20}: {carpeta_salida_pdf / f'{nombre}.pdf'}")
