# -*- coding: utf-8 -*-
"""
Construcción de un Espectro de Respuesta
========================================

Este script genera una gráfica "waterfall" (cascada) que ilustra cómo
se construye un espectro de respuesta de aceleraciones a partir de un
acelerograma real:

  1. Se toma el registro de aceleraciones del sismo.
  2. Para cada periodo T de una serie de osciladores de 1 grado de
     libertad (1 GDL) con razón de amortiguamiento crítico ζ, se resuelve
     la ecuación de movimiento y se obtiene la historia de aceleración
     absoluta de la masa.
  3. Se extrae el valor máximo absoluto de cada historia -> Sa(T).
  4. Se grafican en 3D todas las historias de respuesta (eje periodo T,
     eje tiempo t, eje aceleración a) y, al frente, la envolvente de
     máximos que constituye el espectro de respuesta Sa vs T.

Autor: generado para Diego Andrés Alvarez Marín (UNAL) con Claude
"""

import numpy as np
import matplotlib.pyplot as plt

# DEFINICIONES DE PARÁMETROS
Tmax = 3.0    # periodo máximo de análisis [s]
g    = 981.0  # aceleración de la gravedad [cm/s^2]
ZETA = 0.05   # razón de amortiguamiento: 5%

# LECTURA DEL ACELEROGRAMA
# Archivo de una sola columna con la aceleración del suelo en unidades
# de g, muestreada cada dt = 0.02 s.

# El Centro (18 de mayo de 1940), componente Norte-Sur
# Fuente del registro: NHERI-SimCenter/MDOF (GitHub), componente N-S del
# sismo El Centro 1940, 1560 puntos, dt = 0.02 s, unidades en g
# (PGA verificado = 0.319 g, duración = 31.18 s), consistente con el
# registro históricamente reportado en la literatura de ingeniería sísmica.

ARCHIVO_SISMO = "elcentro_NS.dat"
NOMBRE_SISMO  = "El Centro"
ANIO_SISMO    = 1940
dt            = 0.02                   # paso de tiempo del registro [s]

ag_g = np.loadtxt(ARCHIVO_SISMO)       # aceleración del suelo en [g]
ag = ag_g*g                            # aceleración del suelo en [cm/s^2]
t_sismo = np.arange(len(ag))*dt        # vector de tiempo [s]
tmax_sismo = t_sismo[-1]               # duración del registro [s]

print(f"Registro cargado: {len(ag)} puntos, dt = {dt} s, "
      f"duración = {t_sismo[-1]:.2f} s")
print(f"PGA = {np.max(np.abs(ag_g)):.4f} g "
      f"({np.max(np.abs(ag)):.1f} cm/s^2)")

# SOLUCIÓN DE LA ECUACIÓN DE MOVIMIENTO (MÉTODO DE NEWMARK-BETA)
def respuesta_1gdl_newmark(ag, dt, T, zeta, beta=1/4, gamma=1/2):
    """
    Resuelve la ecuación de movimiento de un oscilador de 1 GDL sometido
    a una aceleración de base ag(t), mediante el método de integración
    directa de Newmark (aceleración promedio constante: beta=1/4,
    gamma=1/2, incondicionalmente estable).

        m*u'' + c*u' + k*u = -m*ag(t)

    Parámetros
    ----------
    ag    : array, aceleración del suelo [cm/s^2]
    dt    : paso de tiempo del registro [s]
    T     : periodo natural del oscilador [s]
    zeta  : razón de amortiguamiento crítico (p.ej. 0.05 para 5%)

    Retorna
    -------
    u, v, a : desplazamiento, velocidad y aceleración RELATIVOS de la masa
    a_abs   : aceleración ABSOLUTA de la masa (a_abs'' = a_rel + ag),
              que es la cantidad físicamente registrada por un
              acelerómetro montado sobre la masa, y la que define el
              espectro de respuesta de aceleraciones Sa.
    """
    wn = 2*np.pi/T                     # frecuencia natural [rad/s]
    m = 1.0                            # masa unitaria (no afecta Sa)
    k = m*wn**2
    c = 2*zeta*m*wn

    n = len(ag)
    u = np.zeros(n)
    v = np.zeros(n)
    a = np.zeros(n)

    # Condiciones iniciales (sistema en reposo)
    a[0] = (-m*ag[0] - c*v[0] - k*u[0])/m

    # Constantes de integración de Newmark
    a1 = m/(beta*dt**2) + gamma*c/(beta*dt)
    a2 = m/(beta*dt) + (gamma/beta - 1)*c
    a3 = (1/(2*beta) - 1)*m + dt*(gamma/(2*beta) - 1)*c
    k_hat = k + a1

    for i in range(n - 1):
        u[i + 1] = ((-m*ag[i + 1]) + a1*u[i] + a2*v[i] + a3*a[i])/k_hat
        v[i + 1] = (gamma/(beta*dt))*(u[i + 1] - u[i]) + (1 - gamma/beta)*v[i] + dt*(1 - gamma/(2*beta))*a[i]
        a[i + 1] = (u[i + 1] - u[i])/(beta*dt**2) - v[i]/(beta*dt) - (1/(2*beta) - 1)*a[i]

    a_abs = a + ag   # aceleración absoluta de la masa
    return u, v, a, a_abs

# SOLUCIÓN DE LA ECUACIÓN DE MOVIMIENTO (INTEGRANDO CON scipy.integrate.solve_ivp)
def respuesta_1gdl_solve_ivp(ag, dt, T, zeta):
    """
    Resuelve la ecuación de movimiento de un oscilador de 1 GDL sometido
    a una aceleración de base ag(t), mediante el método de integración
    directa de scipy.integrate.solve_ivp.

        m*u'' + c*u' + k*u = -m*ag(t)

    Parámetros
    ----------
    ag    : array, aceleración del suelo [cm/s^2]
    dt    : paso de tiempo del registro [s]
    T     : periodo natural del oscilador [s]
    zeta  : razón de amortiguamiento crítico (p.ej. 0.05 para 5%)

    Retorna
    -------
    u, v, a : desplazamiento, velocidad y aceleración RELATIVOS de la masa
    a_abs   : aceleración ABSOLUTA de la masa (a_abs'' = a_rel + ag),
              que es la cantidad físicamente registrada por un
              acelerómetro montado sobre la masa, y la que define el
              espectro de respuesta de aceleraciones Sa.
    """
    from scipy.integrate import solve_ivp

    wn = 2*np.pi/T                     # frecuencia natural [rad/s]
    m = 1.0                            # masa unitaria (no afecta Sa)
    k = m*wn**2
    c = 2*zeta*m*wn

    def ecuacion_movimiento(t, y):
        u, v = y
        # Interpolación lineal para obtener ag(t) en tiempo t
        idx = int(t/dt)
        if idx >= len(ag) - 1:
            ag_t = ag[-1]
        else:
            ag_t = ag[idx] + (ag[idx + 1] - ag[idx]) * ((t - idx*dt)/dt)
        du_dt = v
        dv_dt = (-c*v - k*u - m*ag_t)/m
        return [du_dt, dv_dt]

    t_sismo = np.arange(len(ag))*dt
    t_span = (t_sismo[0], t_sismo[-1])
    sol = solve_ivp(ecuacion_movimiento, t_span, [0, 0], t_eval=t_sismo)

    u = sol.y[0]
    v = sol.y[1]
    a_rel = np.empty_like(u)
    for i, tiempo in enumerate(t_sismo):
        idx = min(int(tiempo/dt), len(ag) - 1)
        if idx == len(ag) - 1:
            ag_t = ag[-1]
        else:
            fraccion = (tiempo - idx*dt)/dt
            ag_t = ag[idx] + (ag[idx + 1] - ag[idx])*fraccion
        a_rel[i] = (-c*v[i] - k*u[i] - m*ag_t)/m
    a_abs = a_rel + ag                # aceleración absoluta de la masa

    return u, v, a_rel, a_abs

# CÁLCULO DEL ESPECTRO
def calcular_espectro_respuesta(ag, dt, Tmax, ZETA):
    """
    Calcula el espectro de respuesta de aceleraciones Sa(T) para un
    rango de periodos T desde 0 hasta Tmax, con una razón de
    amortiguamiento crítico ζ dada.
    """
    periodos_finos = np.arange(0.01, Tmax + 0.01, 0.01)   # para la curva envolvente
    Sa             = np.zeros_like(periodos_finos)        # espectro de respuesta [cm/s^2]

    for i, T in enumerate(periodos_finos):
        _, _, _, a_abs = respuesta_1gdl_newmark(ag, dt, T, ZETA)
        #_, _, _, a_abs = respuesta_1gdl_solve_ivp(ag, dt, T, ZETA)
        Sa[i] = np.max(np.abs(a_abs))

    return periodos_finos, Sa

# DIBUJO DEL ESPECTRO DE RESPUESTA 2D
def plot_espectro_2D(periodos, Sa, ZETA, nombre_sismo, anio_sismo):
    """
    Genera una gráfica 2D del espectro de respuesta de aceleraciones Sa
    vs periodo T, para un amortiguamiento ζ dado.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(periodos, Sa/g, color="black", lw=1.5)
    ax.set_xlabel("Periodo T (s)")
    ax.set_ylabel("Sa (g)")
    ax.set_title(f"Espectro de respuesta de aceleraciones - {nombre_sismo} ({anio_sismo}), ζ={100*ZETA:.0f}%")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(f"espectro_respuesta_{nombre_sismo}_{anio_sismo}.pdf", dpi=200)
    print(f"Figura guardada como 'espectro_respuesta_{nombre_sismo}_{anio_sismo}.pdf'")
    plt.show()

# DIBUJO DEL ESPECTRO DE RESPUESTA 3D (WATERFALL)
def plot_espectro_waterfall(periodos, Sa, t_sismo, ag, ZETA, nombre_sismo, anio_sismo):
    """
    Genera una gráfica 3D tipo "waterfall" (cascada) que ilustra cómo
    se construye un espectro de respuesta de aceleraciones a partir de
    un acelerograma real.
    """
    # Periodos "representativos" que se muestran como historias individuales
    #| en la gráfica de cascada (waterfall)
    periodos_waterfall = [0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    colores = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#e377c2"]

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Historia del sismo (T -> 0, al fondo del eje periodo)
    ax.plot(np.zeros_like(t_sismo), t_sismo, ag, color="black", lw=0.6,
            label=f"Acelerograma {NOMBRE_SISMO} ({ANIO_SISMO})")

    # Historias de respuesta para cada periodo seleccionado
    picos_waterfall = []   # guarda (T, t_del_pico, valor_pico) para las líneas guía
    for T, color in zip(periodos_waterfall, colores):
        _, _, _, a_abs = respuesta_1gdl_newmark(ag, dt, T, ZETA)
        ax.plot(np.full_like(t_sismo, T), t_sismo, a_abs, color=color, lw=0.6)
        idx_pico = np.argmax(np.abs(a_abs))
        picos_waterfall.append((T, t_sismo[idx_pico], np.abs(a_abs[idx_pico])))
        # Etiqueta T=... a un lado del pico de cada traza
        ax.text(T, tmax_sismo, 0, f"T={T}s", color=color, fontsize=8)

    # Curva envolvente del espectro de respuesta (al fondo, t≈0)
    t_fondo = np.full_like(periodos_finos, tmax_sismo)  # se dibuja pegada al fondo

    ax.plot(periodos_finos, t_fondo, Sa, color="black", lw=1.8, label=f"Espectro de respuesta (ζ={ZETA*100:.0f}%)")

    # Líneas guía punteadas: del pico de cada historia al espectro
    for T, t_pico, valor_pico in picos_waterfall:
        idx = np.argmin(np.abs(periodos_finos - T))
        valor_espectro = Sa[idx]
        # Línea punteada negra que conecta el pico (T, t_pico, valor_pico)
        # con el punto correspondiente sobre la envolvente (T, 0, valor_espectro)
        ax.plot([T, T], [t_pico, tmax_sismo], [valor_pico, valor_espectro], color="black", lw=0.7, linestyle='--')
        ax.scatter([T, T], [t_pico, tmax_sismo], [valor_pico, valor_espectro], color="black", s=15, depthshade=False)

    # Formato de ejes y etiquetas
    ax.set_xlabel("Periodo T (s)", labelpad=10)
    ax.set_ylabel("Tiempo (s)", labelpad=10)
    ax.set_zlabel("Aceleración (cm/s²)", labelpad=10)
    ax.set_title("Construcción de un espectro de respuesta\n"
                f"Sismo {NOMBRE_SISMO} ({ANIO_SISMO}) - ζ = {ZETA*100:.0f}%", fontsize=12)

    ax.set_xlim(0, Tmax)
    ax.set_ylim(0, tmax_sismo)
    ax.view_init(elev=18, azim=-61)   # ángulo de cámara similar a la referencia

    # Remueve el fondo de los ejes para que se vea más limpio
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    plt.tight_layout()
    plt.savefig(f"espectro_respuesta_waterfall_{nombre_sismo}_{anio_sismo}.pdf", dpi=200)
    print(f"Figura guardada como 'espectro_respuesta_waterfall_{nombre_sismo}_{anio_sismo}.pdf'")

    plt.show()

# EJECUCIÓN DEL CÁLCULO Y DIBUJO DE GRÁFICAS
periodos_finos, Sa = calcular_espectro_respuesta(ag, dt, Tmax, ZETA)
plot_espectro_2D(periodos_finos, Sa, ZETA, NOMBRE_SISMO, ANIO_SISMO)
plot_espectro_waterfall(periodos_finos, Sa, t_sismo, ag, ZETA, NOMBRE_SISMO, ANIO_SISMO)
