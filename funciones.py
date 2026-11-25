"""
funciones.py - Funciones propias del TFI (Probabilidad y Estadistica con datos reales).

Contiene:
  Parte 1: preparar(df)  -> devuelve df_pob (poblacion limpia)
  Parte 2: media, varianza_pob, mediana, sturges
  Parte 4: bayes, dibujar_arbol
  Parte 5: pmf_binomial, pmf_hipergeometrica, pmf_poisson
  Utilidad: guardar_fig
"""
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuracion de roles
# ---------------------------------------------------------------------------
ROLES = {
    "X": "Duracion_min",    # cuantitativa continua (minutos)
    "W": "n_palabras",      # cuantitativa discreta (palabras del titulo)
    "G": "Modo",            # cualitativa nominal (Menor / Mayor)
    "O": "nivel_energia",   # cualitativa ordinal (cuartiles de Energy)
    "B": "muy_popular",     # evento binario: Popularity > UMBRAL_B
    "U": "Artist",          # unidad de agrupamiento
}

UMBRAL_B = 70
ORDEN_G = ["Menor", "Mayor"]
ORDEN_O = ["Baja", "Media-baja", "Media-alta", "Alta"]
ORDEN_O3 = ["Baja", "Media", "Alta"]
ORDEN_W = ["1 palabra", "2 palabras", "3 palabras", "4 o más"]

# Codigos de "no respuesta / no detectado" identificados en la base (tipicos de Spotify).
# Se convierten a NaN antes de calcular.
CODIGOS_NO_RESPUESTA = {
    "Time_Signature": [0],
    "Tempo": [0],
    "Key": [-1],
    "Duration": [0, -1],
}


# ---------------------------------------------------------------------------
# Parte 1
# ---------------------------------------------------------------------------
def _titulo_limpio(serie):
    """Quita sufijos de version ' - Live ...' y textos entre parentesis del titulo."""
    t = serie.astype(str).str.replace(r"\s[-–]\s.*$", "", regex=True)
    t = t.str.replace(r"\(.*?\)", "", regex=True).str.strip()
    return t


def preparar(df):
    """Aplica TODOS los filtros y recodificaciones y devuelve df_pob."""
    d = df.copy()
    d.columns = d.columns.str.strip()

    # 1) Codigos de no respuesta -> NaN
    for col, codigos in CODIGOS_NO_RESPUESTA.items():
        if col in d.columns:
            d[col] = d[col].replace(codigos, np.nan)

    # 2) Duplicados exactos
    d = d.drop_duplicates()

    # 3) Variables de los roles
    d["Duracion_min"] = d["Duration"] / 60000
    d["n_palabras"] = _titulo_limpio(d["Track"]).str.split().str.len()
    d["Modo"] = d["Mode"].map({0: "Menor", 1: "Mayor"})

    # 4) Filtros de validez (sin imputar: se descartan filas inutilizables)
    d = d.dropna(subset=["Duracion_min", "n_palabras", "Modo", "Energy", "Popularity", "Artist"])
    d = d[(d["Duracion_min"] > 0) & (d["n_palabras"] >= 1)]
    d = d[d["Popularity"].between(0, 100) & d["Energy"].between(0, 1)]
    d = d.reset_index(drop=True)

    # 5) Tipos y variables derivadas
    d["n_palabras"] = d["n_palabras"].astype(int)
    d["Modo"] = pd.Categorical(d["Modo"], categories=ORDEN_G, ordered=False)
    tipo_o = pd.CategoricalDtype(ORDEN_O, ordered=True)
    d["nivel_energia"] = pd.qcut(d["Energy"], q=4, labels=ORDEN_O).astype(tipo_o)
    map_o3 = {"Baja": "Baja", "Media-baja": "Media", "Media-alta": "Media", "Alta": "Alta"}
    d["o_3"] = d["nivel_energia"].astype(str).map(map_o3).astype(
        pd.CategoricalDtype(ORDEN_O3, ordered=True))
    d["w_grupo"] = pd.cut(d["n_palabras"], bins=[0, 1, 2, 3, np.inf], labels=ORDEN_W)
    d["muy_popular"] = d["Popularity"] > UMBRAL_B

    # 6) Validaciones
    for rol, col in ROLES.items():
        assert d[col].notna().all(), f"{rol} ({col}) tiene vacios"
    assert (d[ROLES["X"]] > 0).all(), "X tiene valores imposibles"
    assert d["nivel_energia"].cat.ordered and d["o_3"].cat.ordered, "O debe ser category ordenada"
    assert d["muy_popular"].dtype == bool, "B debe ser booleano"
    assert 0.02 <= d["muy_popular"].mean() <= 0.50, "La proporcion de B debe estar entre 2% y 50%"
    assert d["w_grupo"].notna().all(), "w_grupo con vacios"
    return d


# ---------------------------------------------------------------------------
# Parte 2 - estadisticos propios (sin pandas/numpy)
# ---------------------------------------------------------------------------
def media(x):
    x = list(x)
    return sum(x) / len(x)


def varianza_pob(x):
    x = list(x)
    m = media(x)
    return sum((v - m) ** 2 for v in x) / len(x)  # ddof = 0


def mediana(x):
    s = sorted(x)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 == 1 else (s[mid - 1] + s[mid]) / 2


def sturges(n):
    """Numero de clases (sin redondear): k = 1 + 3,322 * log10(N)."""
    return 1 + 3.322 * math.log10(n)


# ---------------------------------------------------------------------------
# Parte 4 - Bayes
# ---------------------------------------------------------------------------
def bayes(priori, verosimilitud):
    """Devuelve P(Ei | B) a partir de P(Ei) y P(B | Ei)."""
    indice = priori.index if isinstance(priori, pd.Series) else None
    pri = np.asarray(priori, dtype=float)
    ver = np.asarray(verosimilitud, dtype=float)
    conjunta = pri * ver
    post = conjunta / conjunta.sum()  # el denominador es P(B) por probabilidad total
    return pd.Series(post, index=indice) if indice is not None else list(post)


def dibujar_arbol(categorias, priori, verosim, titulo="", ruta=None):
    """Diagrama de arbol: raiz -> categorias Ei -> (B / no B). Devuelve las hojas."""
    n = len(categorias)
    fig, ax = plt.subplots(figsize=(10, 1.6 * n + 1.5))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ys_hojas = np.linspace(0.95, 0.05, 2 * n)
    hojas = []
    for i, (cat, pi, vi) in enumerate(zip(categorias, priori, verosim)):
        y_hoja_b, y_hoja_nb = ys_hojas[2 * i], ys_hojas[2 * i + 1]
        y_nodo = (y_hoja_b + y_hoja_nb) / 2
        ax.plot([0.04, 0.34], [0.5, y_nodo], color="gray")
        ax.text(0.19, (0.5 + y_nodo) / 2 + 0.015, f"P = {pi:.3f}", fontsize=9)
        ax.text(0.34, y_nodo, f" {cat}", fontsize=10, fontweight="bold", va="center")
        for y_h, etiqueta, prob in ((y_hoja_b, "B", vi), (y_hoja_nb, "no B", 1 - vi)):
            ax.plot([0.46, 0.66], [y_nodo, y_h], color="gray")
            ax.text(0.54, (y_nodo + y_h) / 2 + 0.012, f"{prob:.3f}", fontsize=9)
            camino = pi * prob
            ax.text(0.67, y_h, f"{etiqueta}:  P = {camino:.4f}", fontsize=9, va="center")
            hojas.append(camino)
    ax.plot(0.04, 0.5, "ko")
    ax.set_title(titulo)
    if ruta:
        fig.tight_layout()
        fig.savefig(ruta, dpi=150)
    plt.show()
    return hojas


# ---------------------------------------------------------------------------
# Parte 5 - modelos de probabilidad "a mano"
# ---------------------------------------------------------------------------
def pmf_binomial(k, n, p):
    return math.comb(n, k) * p ** k * (1 - p) ** (n - k)


def pmf_hipergeometrica(k, N, K, n):
    """P(X = k): poblacion N, K exitos, muestra n sin reposicion."""
    if k < 0 or k > K or n - k > N - K or n - k < 0:
        return 0.0
    return math.comb(K, k) * math.comb(N - K, n - k) / math.comb(N, n)


def pmf_poisson(k, lam):
    return math.exp(-lam) * lam ** k / math.factorial(k)


# ---------------------------------------------------------------------------
# Utilidad
# ---------------------------------------------------------------------------
def guardar_fig(nombre, fuente, carpeta="figuras"):
    """Agrega la fuente al pie, guarda con dpi=150 (sin capturas) y muestra el grafico."""
    os.makedirs(carpeta, exist_ok=True)
    fig = plt.gcf()
    fig.text(0.01, 0.005, fuente, fontsize=7, style="italic", ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(os.path.join(carpeta, f"{nombre}.png"), dpi=150)
    plt.show()
