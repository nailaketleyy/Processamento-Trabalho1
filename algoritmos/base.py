"""Funções auxiliares compartilhadas por todas as etapas."""

import numpy as np

L = 256  # número de níveis de intensidade (imagens de 8 bits)


def cinza(img):
    """Converte RGB para tons de cinza (luminância) e devolve matriz 2D."""
    g = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    return np.clip(g, 0, 255).astype(np.uint8)


def cinza_para_rgb(g):
    """Replica uma matriz 2D nos três canais."""
    return np.stack([g, g, g], axis=-1)


def aplicar_lut(img, lut):
    """Aplica uma tabela de consulta (lookup table) a todos os canais.
    img[y, x, c] -> lut[ img[y, x, c] ]"""
    return lut[img]


def ajustar_para_tela(img, larg_max, alt_max):
    """Reduz a imagem (vizinho mais próximo) para caber na área de exibição.
    Usada apenas para visualização; não altera a imagem processada."""
    h, w = img.shape[:2]
    esc = min(larg_max / w, alt_max / h, 1.0)
    nh, nw = max(1, int(h * esc)), max(1, int(w * esc))
    ys = (np.arange(nh) * h / nh).astype(int)
    xs = (np.arange(nw) * w / nw).astype(int)
    return img[ys][:, xs]
