"""Conversões e operações de cor RGB/HSI para imagens uint8."""

import numpy as np

from .base import cinza
from .histograma import equalizar_cinza


def rgb_para_hsi(img):
    """Converte RGB uint8 em HSI: H em graus, S e I no intervalo [0, 1]."""
    rgb = np.asarray(img, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("A imagem RGB deve ter três canais.")
    r, g, b = (rgb[..., i] / 255.0 for i in range(3))
    soma = r + g + b
    minimo = np.minimum(np.minimum(r, g), b)
    intensidade = soma / 3.0
    saturacao = np.ones_like(intensidade)
    np.divide(3.0 * minimo, soma, out=saturacao, where=soma > 0)
    saturacao = 1.0 - saturacao
    numerador = 0.5 * ((r - g) + (r - b))
    denominador = np.sqrt((r - g) ** 2 + (r - b) * (g - b))
    razao = np.zeros_like(intensidade)
    np.divide(numerador, denominador, out=razao, where=denominador > 0)
    theta = np.arccos(np.clip(razao, -1.0, 1.0))
    matiz = np.where(b <= g, theta, 2 * np.pi - theta)
    matiz = np.where(saturacao == 0, 0.0, matiz)
    return np.rad2deg(matiz), np.clip(saturacao, 0, 1), np.clip(intensidade, 0, 1)


def hsi_para_rgb(matiz, saturacao, intensidade):
    """Converte HSI (H em graus, S/I em [0,1]) para uint8 RGB."""
    h, s, i = np.broadcast_arrays(
        np.asarray(matiz, dtype=np.float64),
        np.asarray(saturacao, dtype=np.float64),
        np.asarray(intensidade, dtype=np.float64),
    )
    if not np.isfinite(h).all() or not np.isfinite(s).all() or not np.isfinite(i).all():
        raise ValueError("Os componentes HSI devem ser finitos.")
    h = np.mod(h, 360.0)
    s = np.clip(s, 0.0, 1.0)
    i = np.clip(i, 0.0, 1.0)
    r = np.empty_like(i)
    g = np.empty_like(i)
    b = np.empty_like(i)
    setores = (
        (h < 120.0, h),
        ((h >= 120.0) & (h < 240.0), h - 120.0),
        (h >= 240.0, h - 240.0),
    )
    for indice, (mascara, local) in enumerate(setores):
        denominador = np.cos(np.deg2rad(60.0 - local))
        razao = np.zeros_like(i)
        np.divide(
            s * np.cos(np.deg2rad(local)), denominador,
            out=razao, where=np.abs(denominador) > 1e-12,
        )
        dominante = i * (1.0 + razao)
        baixo = i * (1.0 - s)
        terceiro = 3.0 * i - dominante - baixo
        if indice == 0:
            r[mascara], g[mascara], b[mascara] = (
                dominante[mascara], terceiro[mascara], baixo[mascara])
        elif indice == 1:
            r[mascara], g[mascara], b[mascara] = (
                baixo[mascara], dominante[mascara], terceiro[mascara])
        else:
            r[mascara], g[mascara], b[mascara] = (
                terceiro[mascara], baixo[mascara], dominante[mascara])
    rgb = np.stack((r, g, b), axis=-1)
    return np.clip(np.floor(rgb * 255.0 + 0.5), 0, 255).astype(np.uint8)


def componente_hsi(img, componente):
    """Visualiza H, S ou I como uma imagem RGB em tons de cinza."""
    h, s, i = rgb_para_hsi(img)
    if componente == "H":
        valor = h / 360.0
    elif componente == "S":
        valor = s
    elif componente == "I":
        valor = i
    else:
        raise ValueError("Componente HSI deve ser 'H', 'S' ou 'I'.")
    canal = np.clip(np.floor(valor * 255 + 0.5), 0, 255).astype(np.uint8)
    return np.repeat(canal[..., None], 3, axis=2)


def escala_cinza(img, metodo="ponderada"):
    """Converte RGB para cinza por média aritmética ou luminância ponderada."""
    rgb = np.asarray(img)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("A imagem deve ser RGB.")
    if metodo == "media":
        gray = np.floor(rgb.astype(np.float64).mean(axis=2) + 0.5).astype(np.uint8)
    elif metodo == "ponderada":
        gray = cinza(rgb)
    else:
        raise ValueError("Método de cinza deve ser 'media' ou 'ponderada'.")
    return np.repeat(gray[..., None], 3, axis=2)


def negativo(img):
    return 255 - np.asarray(img, dtype=np.uint8)


def equalizar_hsi(img):
    """Equaliza o componente I do modelo HSI e reconverte para RGB."""
    h, s, intensidade = rgb_para_hsi(img)
    i8 = np.clip(np.floor(intensidade * 255 + 0.5), 0, 255).astype(np.uint8)
    ieq = equalizar_cinza(i8).astype(np.float64) / 255.0
    return hsi_para_rgb(h, s, ieq)


def ajustar_hsi(img, delta_h=0.0, fator_s=1.0, fator_i=1.0):
    """Ajusta matiz em graus e multiplica saturação e intensidade."""
    if not np.isfinite([delta_h, fator_s, fator_i]).all():
        raise ValueError("Os ajustes HSI devem ser valores finitos.")
    h, s, i = rgb_para_hsi(img)
    return hsi_para_rgb(h + delta_h, s * fator_s, i * fator_i)


def ajustar_cmy(img, cyan_red=0.0, magenta_green=0.0, yellow_blue=0.0):
    """Ajusta os pares C/R, M/G e Y/B como offsets assinados dos canais RGB."""
    ajustes = (float(cyan_red), float(magenta_green), float(yellow_blue))
    if not np.isfinite(ajustes).all():
        raise ValueError("Os ajustes C/R, M/G e Y/B devem ser finitos.")
    rgb = np.asarray(img, dtype=np.float64).copy()
    rgb[..., 0] += ajustes[0]
    rgb[..., 1] += ajustes[1]
    rgb[..., 2] += ajustes[2]
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)


def sepia(img):
    """Converte para cinza e aplica uma tonalidade sépia."""
    gray = cinza(img).astype(np.float64)
    r = np.minimum(255, gray * 1.07 + 20)
    g = np.minimum(255, gray * 0.96 + 10)
    b = gray * 0.78
    return np.clip(np.floor(np.stack((r, g, b), axis=-1) + 0.5), 0, 255).astype(np.uint8)


def chroma_key(img, substituta, distancia=100.0):
    """Substitui pixels verdes por uma imagem RGB redimensionada por nearest."""
    frente = np.asarray(img, dtype=np.uint8)
    fundo = np.asarray(substituta, dtype=np.uint8)
    if frente.ndim != 3 or frente.shape[2] != 3 or fundo.ndim != 3 or fundo.shape[2] != 3:
        raise ValueError("As imagens de chroma-key devem ser RGB.")
    if not np.isfinite(distancia) or distancia < 0:
        raise ValueError("A distância da cor deve ser finita e não negativa.")
    h, w = frente.shape[:2]
    ys = np.minimum((np.arange(h) * fundo.shape[0] / h).astype(int), fundo.shape[0] - 1)
    xs = np.minimum((np.arange(w) * fundo.shape[1] / w).astype(int), fundo.shape[1] - 1)
    fundo_redim = fundo[ys][:, xs]
    verde = np.array([0.0, 255.0, 0.0])
    dist = np.linalg.norm(frente.astype(np.float64) - verde, axis=2)
    mascara = dist <= distancia
    resultado = frente.copy()
    resultado[mascara] = fundo_redim[mascara]
    return resultado
