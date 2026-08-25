# Este programa genera los acelerogramas (g) a partir de los .dat
# que arrojaron los acelerómetros. El programa corrije por línea
# base usando una corrección lineal.

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import detrend

# Archivo ODS con los metadatos de las estaciones (nombre de carpeta,
# coordenadas, horas UTC y factores de conversión de cada componente).
ARCHIVO_ODS = "sismo_ago_10_2026.ods"
NOMBRE_COLUMNA = "Filename"
COLUMNA_FS = "Frec_muestreo_Hz"
NUMERO_COMPONENTES = 3  # vertical, N-S y E-O
COLUMNAS_UV_POR_BIT = ["uV/bit_1", "uV/bit_2", "uV/bit_3"]
COLUMNAS_V_POR_G = ["V/g_1", "V/g_2", "V/g_3"]


# Parámetros para recortar la ventana del acelerograma alrededor del sismo
VENTANA_PRE_SISMO_S = 30       # segundos que se dejan antes del inicio detectado
VENTANA_TOTAL_S = 4 * 60       # duración total de la ventana extraída
UMBRAL_ACELERACION_G = 0.0005  # umbral (g) del promedio móvil para detectar el inicio

# Lee el archivo ODS con los metadatos de las estaciones
ruta_ods = Path(ARCHIVO_ODS).resolve()
carpeta_base = ruta_ods.parent
df = pd.read_excel(ruta_ods, engine="odf")

if NOMBRE_COLUMNA not in df.columns:
    raise KeyError(
        f"El archivo no contiene la columna requerida: {NOMBRE_COLUMNA}"
    )

if COLUMNA_FS not in df.columns:
    raise KeyError(
        f"El archivo no contiene la columna requerida: {COLUMNA_FS}"
    )

# Verifica que en el archivo ODS estén las columnas de corrección de los .dat
columnas_factores = COLUMNAS_UV_POR_BIT + COLUMNAS_V_POR_G
columnas_faltantes = [
    columna for columna in columnas_factores if columna not in df.columns
]
if columnas_faltantes:
    raise KeyError(
        f"El archivo no contiene las columnas de conversión requeridas: "
        f"{', '.join(columnas_faltantes)}"
    )

# Factores de conversión de cada estación, indexados por nombre de carpeta.
factores = df.set_index(NOMBRE_COLUMNA)[columnas_factores]
# Frecuencia de muestreo de cada estación, indexada por nombre de carpeta.
frecuencias_muestreo = df.set_index(NOMBRE_COLUMNA)[COLUMNA_FS]

# Crea las carpetas de salida
carpeta_salida_npy = carpeta_base / "acelerogramas_npy"
carpeta_salida_npy.mkdir(exist_ok=True)
carpeta_salida_txt = carpeta_base / "acelerogramas_txt"
carpeta_salida_txt.mkdir(exist_ok=True)

# Recorre cada carpeta para generar los acelerogramas
for nombre_carpeta in df[NOMBRE_COLUMNA].dropna().astype(str):
    # Lee los archivos .dat dentro de cada carpeta
    carpeta = carpeta_base / "dats" / nombre_carpeta.strip()
    archivos_dat = sorted(carpeta.glob("*.dat"))  # orden: C1, C2, C3

    # Verifica que haya exactamente 3 archivos .dat en la carpeta
    if len(archivos_dat) != NUMERO_COMPONENTES:
        raise ValueError(
            f"{carpeta}: se esperaban {NUMERO_COMPONENTES} archivos .dat, "
            f"pero se encontraron {len(archivos_dat)}"
        )

    # Cada .dat trae 2 filas de encabezado (número de datos y fs) antes
    # de la serie de aceleración en bits crudos del sensor; no se requieren.
    columnas = [np.loadtxt(archivo, skiprows=2) for archivo in archivos_dat]
    longitudes = {columna.size for columna in columnas}
    if len(longitudes) != 1:
        raise ValueError(
            f"Los archivos de {carpeta} tienen longitudes diferentes"
        )

    # Obtiene los factores de conversión para la carpeta actual
    try:
        factores_carpeta = factores.loc[nombre_carpeta.strip()]
    except KeyError as error:
        raise KeyError(
            f"No se encontraron factores de conversión para {nombre_carpeta.strip()}"
        ) from error

    factores_uv_por_bit = factores_carpeta[COLUMNAS_UV_POR_BIT].to_numpy(dtype=float)
    factores_v_por_g    = factores_carpeta[COLUMNAS_V_POR_G].to_numpy(dtype=float)
    if np.any(factores_v_por_g == 0):
        raise ValueError(f"El factor V/g de {nombre_carpeta.strip()} no puede ser cero")

    fs = float(frecuencias_muestreo.loc[nombre_carpeta.strip()])  # frecuencia de muestreo [Hz]

    # Conversión bits -> voltios (uV/bit / 1e6) -> aceleración en g (/ V/g).
    matriz_bits = np.column_stack(columnas)
    matriz = matriz_bits * factores_uv_por_bit / 1_000_000 / factores_v_por_g

    # Corrección de línea base: remueve la tendencia lineal de cada componente.
    matriz = detrend(matriz, axis=0, type="linear")

    # Detecta el inicio del sismo: el promedio móvil (ventana de 30 s) de la
    # magnitud de aceleración supera el umbral por primera vez en ese punto.
    muestras_pre   = int(round(VENTANA_PRE_SISMO_S * fs))
    muestras_total = int(round(VENTANA_TOTAL_S * fs))
    magnitud = np.sqrt(np.sum(matriz**2, axis=1))
    promedio_movil = (
        pd.Series(magnitud)
        .rolling(window=muestras_pre, min_periods=muestras_pre)
        .mean()
        .to_numpy()
    )
    indices_sobre_umbral = np.flatnonzero(promedio_movil > UMBRAL_ACELERACION_G)

    if indices_sobre_umbral.size == 0:
        print(f"Advertencia: {nombre_carpeta.strip()} no superó el umbral; se conserva el registro completo")
        inicio_ventana, fin_ventana = 0, matriz.shape[0]
    else:
        inicio_sismo = indices_sobre_umbral[0]
        inicio_ventana = max(0, inicio_sismo - muestras_pre)
        fin_ventana = min(matriz.shape[0], inicio_ventana + muestras_total)
    matriz = matriz[inicio_ventana:fin_ventana]

    # Guarda los acelerogramas en formato .npy y .txt
    np.save(carpeta_salida_npy / f"{nombre_carpeta.strip()}.npy", matriz)
    np.savetxt(carpeta_salida_txt / f"{nombre_carpeta.strip()}.txt", matriz, delimiter="\t")
    print(f"{nombre_carpeta.strip():<20}: {matriz.shape} <- "
            f"{', '.join(archivo.name for archivo in archivos_dat)}")
