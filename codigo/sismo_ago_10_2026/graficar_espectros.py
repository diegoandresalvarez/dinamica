from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CARPETA_NP = "acelerogramas_np"
DT = 1 / 200.0
ZETA = 0.05
T_MAX = 3.0
DELTA_T = 0.01
COMPONENTES = [
	"Vertical",
	"Horizontal Norte-Sur",
	"Horizontal Este-Oeste",
]


def respuesta_1gdl_newmark(aceleracion_suelo, dt, periodo, zeta, beta=1 / 4, gamma=1 / 2):
	"""Resuelve la respuesta de un oscilador de 1 GDL mediante Newmark."""
	frecuencia_natural = 2 * np.pi / periodo
	masa = 1.0
	rigidez = masa * frecuencia_natural**2
	amortiguamiento = 2 * zeta * masa * frecuencia_natural

	n = len(aceleracion_suelo)
	desplazamiento = np.zeros(n)
	velocidad = np.zeros(n)
	aceleracion_relativa = np.zeros(n)

	aceleracion_relativa[0] = (
		-masa * aceleracion_suelo[0] - amortiguamiento * velocidad[0] - rigidez * desplazamiento[0]
	) / masa

	a1 = masa / (beta * dt**2) + gamma * amortiguamiento / (beta * dt)
	a2 = masa / (beta * dt) + (gamma / beta - 1) * amortiguamiento
	a3 = (1 / (2 * beta) - 1) * masa + dt * (gamma / (2 * beta) - 1) * amortiguamiento
	rigidez_efectiva = rigidez + a1

	for indice in range(n - 1):
		desplazamiento[indice + 1] = (
			(-masa * aceleracion_suelo[indice + 1])
			+ a1 * desplazamiento[indice]
			+ a2 * velocidad[indice]
			+ a3 * aceleracion_relativa[indice]
		) / rigidez_efectiva
		velocidad[indice + 1] = (
			(gamma / (beta * dt)) * (desplazamiento[indice + 1] - desplazamiento[indice])
			+ (1 - gamma / beta) * velocidad[indice]
			+ dt * (1 - gamma / (2 * beta)) * aceleracion_relativa[indice]
		)
		aceleracion_relativa[indice + 1] = (
			(desplazamiento[indice + 1] - desplazamiento[indice]) / (beta * dt**2)
			- velocidad[indice] / (beta * dt)
			- (1 / (2 * beta) - 1) * aceleracion_relativa[indice]
		)

	aceleracion_absoluta = aceleracion_relativa + aceleracion_suelo
	return aceleracion_absoluta


def calcular_espectro_respuesta(aceleracion_suelo, dt, t_max, zeta):
	"""Calcula el espectro de respuesta Sa(T) para un registro dado."""
	periodos = np.arange(0.01, t_max + DELTA_T, DELTA_T)
	sa = np.zeros_like(periodos)

	for indice, periodo in enumerate(periodos):
		aceleracion_absoluta = respuesta_1gdl_newmark(aceleracion_suelo, dt, periodo, zeta)
		sa[indice] = np.max(np.abs(aceleracion_absoluta))

	return periodos, sa


def graficar_espectros(carpeta_np=CARPETA_NP):
	"""Lee todos los .npy en la carpeta y genera un PDF por estación."""
	ruta_carpeta = Path(carpeta_np).resolve()
	archivos_npy = sorted(ruta_carpeta.glob("*.npy"))

	for archivo in archivos_npy:
		acelerogramas = np.load(archivo)
		if acelerogramas.ndim != 2 or acelerogramas.shape[1] != 3:
			raise ValueError(f"{archivo.name} no tiene forma (n, 3)")

		fig, ejes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
		for indice, componente in enumerate(COMPONENTES):
			periodos, sa = calcular_espectro_respuesta(acelerogramas[:, indice], DT, T_MAX, ZETA)
			ejes[indice].plot(periodos, sa, color="black", lw=1.4)
			ejes[indice].set_title(f"Espectro de respuesta - {componente}")
			ejes[indice].set_ylabel("Sa [g]")
			ejes[indice].grid(True)

		ejes[-1].set_xlabel("Periodo T [s]")
		fig.suptitle(f"Espectros de respuesta de aceleración - {archivo.stem}")
		fig.tight_layout()
		archivo_salida = ruta_carpeta / f"espectro_respuesta_{archivo.stem}.pdf"
		fig.savefig(archivo_salida, dpi=200)
		plt.close(fig)
		print(f"{archivo.stem}: figura guardada en {archivo_salida}")


if __name__ == "__main__":
	graficar_espectros()
