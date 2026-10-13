"""ETAPA 2 (Cap. 3, seções 3.2 e 3.3) - Histograma e equalização de histograma.

Histograma:        h(r_k) = n_k                      (nº de pixels com intensidade r_k)
Normalizado:       p(r_k) = n_k / (M*N)
Equalização:       s_k = T(r_k) = (L-1) * soma_{j<=k} p(r_j)      (Eq. 3.3-8)
Média e variância: Eq. 3.3-18 e 3.3-21 (calculadas a partir do histograma)
"""

import numpy as np

from .base import cinza, cinza_para_rgb


def histograma(canal):
    """Histograma (256 posições) de uma matriz 2D de inteiros em [0, 255]."""
    return np.bincount(canal.reshape(-1), minlength=256)


def histograma_normalizado(h):
    """p(r_k) = n_k / (M*N)"""
    return h / h.sum()


def estatisticas(h):
    """Média e desvio padrão das intensidades, calculados a partir do histograma."""
    p = histograma_normalizado(h)
    r = np.arange(len(h))
    media = float((r * p).sum())
    variancia = float((((r - media) ** 2) * p).sum())
    return media, variancia ** 0.5


def lut_equalizacao(h):
    """Tabela de mapeamento s_k = (L-1) * soma acumulada de p(r_j), arredondada
    para o inteiro mais próximo (Eq. 3.3-8). L = len(h)."""
    p = histograma_normalizado(h)
    acumulada = np.cumsum(p)
    return np.floor((len(h) - 1) * acumulada + 0.5).astype(np.uint8)


def equalizar_cinza(g):
    """Equaliza uma matriz 2D (tons de cinza)."""
    return lut_equalizacao(histograma(g))[g]


def equalizar_canais(img):
    """Equaliza R, G e B separadamente (pode alterar as cores)."""
    saida = np.empty_like(img)
    for c in range(3):
        saida[..., c] = equalizar_cinza(img[..., c])
    return saida


def equalizar_luminancia(img):
    """Equaliza apenas a luminância e soma a mesma variação aos três canais,
    o que mantém a crominância (Cb, Cr) inalterada e preserva as cores."""
    y = cinza(img)
    y_eq = equalizar_cinza(y)
    delta = y_eq.astype(np.int16) - y.astype(np.int16)
    return np.clip(img.astype(np.int16) + delta[..., None], 0, 255).astype(np.uint8)


def equalizar(img, modo="lum"):
    """modo: 'lum' (luminância), 'rgb' (canal a canal) ou 'cinza' (converte p/ cinza)."""
    if modo == "rgb":
        return equalizar_canais(img)
    if modo == "cinza":
        return cinza_para_rgb(equalizar_cinza(cinza(img)))
    return equalizar_luminancia(img)
