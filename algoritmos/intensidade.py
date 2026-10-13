"""ETAPA 1 (Cap. 3, seções 3.1 e 3.2) - Transformações de intensidade.

Negativo, brilho, potência (gama), limiarização, funções lineares definidas
por partes (alargamento de contraste, fatiamento de níveis) e planos de bits.
"""

import numpy as np

from .base import L, cinza, cinza_para_rgb


# ---- Funções que geram a LUT (tabela de 256 entradas) ----

def lut_negativo():
    """s = L - 1 - r   (Eq. 3.2-1)"""
    return np.array([L - 1 - r for r in range(L)], dtype=np.uint8)


def lut_brilho(b):
    """s = r + b  (saturando em [0, 255])"""
    return np.array([min(255, max(0, r + b)) for r in range(L)], dtype=np.uint8)


def lut_gama(c, gama):
    """s = c * r^gama   (Eq. 3.2-3), com r normalizado em [0, 1]
    e resultado reescalado para [0, 255]."""
    lut = np.zeros(L, dtype=np.uint8)
    for r in range(L):
        s = c * ((r / (L - 1)) ** gama) * (L - 1)
        lut[r] = int(min(255, max(0, round(s))))
    return lut


def lut_linear_partes(pontos):
    """Função linear definida por partes que passa pelos pontos (r, s),
    em ordem crescente de r. Dois pontos com o mesmo r criam um degrau."""
    lut = np.zeros(L, dtype=np.uint8)
    for i in range(len(pontos) - 1):
        x0, y0 = pontos[i]
        x1, y1 = pontos[i + 1]
        for r in range(x0, x1 + 1):
            if x1 == x0:
                s = y1
            else:
                s = y0 + (y1 - y0) * (r - x0) / (x1 - x0)
            lut[r] = int(min(255, max(0, round(s))))
    return lut


def lut_fatiamento(a, b, preservar, valor_alto=255, valor_baixo=10):
    """Fatiamento de níveis de intensidade (Fig. 3.11).
    preservar=False -> faixa [a, b] = valor_alto, resto = valor_baixo (Fig. 3.11a)
    preservar=True  -> faixa [a, b] = valor_alto, resto inalterado    (Fig. 3.11b)"""
    lut = np.zeros(L, dtype=np.uint8)
    for r in range(L):
        if a <= r <= b:
            lut[r] = valor_alto
        else:
            lut[r] = r if preservar else valor_baixo
    return lut


# ---- Transformações que não são LUT simples por canal ----

def limiarizar(img, T):
    """Imagem binária: 255 se intensidade >= T, senão 0 (sobre o tom de cinza)."""
    g = cinza(img)
    saida = np.zeros_like(g)
    saida[g >= T] = 255
    return cinza_para_rgb(saida)


def plano_de_bits(img, n):
    """Extrai o n-ésimo plano de bits (1 = menos significativo ... 8 = mais)."""
    g = cinza(img)
    plano = ((g >> (n - 1)) & 1) * 255
    return cinza_para_rgb(plano.astype(np.uint8))
