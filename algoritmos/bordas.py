"""Detectores de borda espaciais baseados nos operadores de Sobel."""

import numpy as np

from .base import cinza, cinza_para_rgb
from .convolucao import SOBEL_X, SOBEL_Y, correlacao, para_uint8


def componentes_sobel(img, borda="replicar"):
    """Calcula as respostas direcionais horizontal e vertical da luminância."""
    g = cinza(img).astype(np.float64)
    gx = correlacao(g, SOBEL_X, borda)
    gy = correlacao(g, SOBEL_Y, borda)
    return gx, gy


def sobel(img, direcao="x", borda="replicar"):
    """Retorna a resposta X ou Y normalizada para uma imagem RGB em tons de cinza."""
    gx, gy = componentes_sobel(img, borda)
    if direcao == "x":
        resposta = gx
    elif direcao == "y":
        resposta = gy
    else:
        raise ValueError("A direção Sobel deve ser 'x' ou 'y'.")
    return cinza_para_rgb(para_uint8(resposta, "normalizar"))


def gradiente(img, borda="replicar"):
    """Magnitude não linear do gradiente: sqrt(Gx² + Gy²), normalizada."""
    gx, gy = componentes_sobel(img, borda)
    magnitude = np.hypot(gx, gy)
    return cinza_para_rgb(para_uint8(magnitude, "normalizar"))
