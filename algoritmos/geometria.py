"""Transformações geométricas inversas com vizinho mais próximo ou bilinear.

As coordenadas são expressas como (x, y), com origem no canto superior
esquerdo. As transformações mantêm o tamanho da imagem e preenchem de preto
as regiões sem correspondência na imagem de entrada.
"""

import numpy as np


def transformar(
    img,
    escala=(1.0, 1.0),
    angulo=0.0,
    deslocamento=(0.0, 0.0),
    pivo=None,
    interpolacao="vizinho",
    expandir=False,
    max_pixels=16_000_000,
):
    """Aplica escala, rotação e translação, nessa ordem geométrica direta.

    `angulo` é em graus e positivo no sentido horário da tela. `pivo` e
    `deslocamento` seguem (x, y). A saída conserva o formato, a menos que
    `expandir=True`; nesse modo, dimensões acima de `max_pixels` são rejeitadas.
    """
    img = np.asarray(img)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D ou 3D não vazia.")
    sx, sy = map(float, escala)
    dx, dy = map(float, deslocamento)
    angulo = float(angulo)
    if not np.isfinite([sx, sy, dx, dy, angulo]).all() or sx <= 0 or sy <= 0:
        raise ValueError("Escalas devem ser positivas e parâmetros finitos.")
    if interpolacao not in ("vizinho", "bilinear"):
        raise ValueError("Interpolação deve ser 'vizinho' ou 'bilinear'.")
    if not isinstance(max_pixels, (int, np.integer)) or max_pixels < 1:
        raise ValueError("O limite de pixels deve ser um inteiro positivo.")

    h, w = img.shape[:2]
    if pivo is None:
        px, py = (w - 1) / 2, (h - 1) / 2
    else:
        px, py = map(float, pivo)
        if not np.isfinite([px, py]).all():
            raise ValueError("O pivô deve conter coordenadas finitas.")

    theta = np.deg2rad(angulo)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if expandir:
        cantos_x = np.array([0, w - 1, 0, w - 1], dtype=np.float64) - px
        cantos_y = np.array([0, 0, h - 1, h - 1], dtype=np.float64) - py
        destinos_x = px + dx + cos_t * sx * cantos_x - sin_t * sy * cantos_y
        destinos_y = py + dy + sin_t * sx * cantos_x + cos_t * sy * cantos_y
        origem_x, origem_y = np.floor(destinos_x.min()), np.floor(destinos_y.min())
        fim_x, fim_y = np.ceil(destinos_x.max()), np.ceil(destinos_y.max())
        largura_saida = int(fim_x - origem_x + 1)
        altura_saida = int(fim_y - origem_y + 1)
    else:
        origem_x = origem_y = 0
        largura_saida, altura_saida = w, h

    total_pixels = largura_saida * altura_saida
    if total_pixels > max_pixels:
        raise ValueError(
            f"A saída de {largura_saida}×{altura_saida} pixels ({total_pixels:,} pixels) "
            f"excede o limite de {max_pixels:,} pixels. Reduza a escala ou desative "
            "'Expandir saída'.")

    yy, xx = np.indices((altura_saida, largura_saida), dtype=np.float64)
    x = xx + origem_x - px - dx
    y = yy + origem_y - py - dy

    # Inverte primeiro a rotação e depois a escala da transformação direta.
    x_src = (cos_t * x + sin_t * y) / sx + px
    y_src = (-sin_t * x + cos_t * y) / sy + py
    dentro = ((x_src >= 0) & (x_src <= w - 1) &
              (y_src >= 0) & (y_src <= h - 1))

    if interpolacao == "vizinho":
        xi = np.floor(x_src + 0.5).astype(np.intp)
        yi = np.floor(y_src + 0.5).astype(np.intp)
        xi = np.clip(xi, 0, w - 1)
        yi = np.clip(yi, 0, h - 1)
        resultado = img[yi, xi].copy()
    else:
        x0 = np.floor(x_src).astype(np.intp)
        y0 = np.floor(y_src).astype(np.intp)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)
        x0 = np.clip(x0, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        wx = (x_src - x0)[..., None] if img.ndim == 3 else (x_src - x0)
        wy = (y_src - y0)[..., None] if img.ndim == 3 else (y_src - y0)
        p00 = img[y0, x0].astype(np.float64)
        p10 = img[y0, x1].astype(np.float64)
        p01 = img[y1, x0].astype(np.float64)
        p11 = img[y1, x1].astype(np.float64)
        resultado = ((1 - wx) * (1 - wy) * p00 +
                     wx * (1 - wy) * p10 +
                     (1 - wx) * wy * p01 +
                     wx * wy * p11)
        resultado = np.clip(np.floor(resultado + 0.5), 0, 255).astype(img.dtype)

    resultado[~dentro] = 0
    return resultado


def _amostrar(img, x_src, y_src, interpolacao):
    """Amostra img nas coordenadas (x_src, y_src) (base 0, float).
    Pontos fora da imagem ficam pretos. Funciona em 2D e 3D."""
    h, w = img.shape[:2]
    dentro = (x_src >= -0.5) & (x_src <= w - 0.5) & (y_src >= -0.5) & (y_src <= h - 0.5)
    if interpolacao == "vizinho":
        xi = np.clip(np.floor(x_src + 0.5).astype(np.intp), 0, w - 1)
        yi = np.clip(np.floor(y_src + 0.5).astype(np.intp), 0, h - 1)
        res = img[yi, xi].copy()
    else:
        xs = np.clip(x_src, 0, w - 1)
        ys = np.clip(y_src, 0, h - 1)
        x0 = np.floor(xs).astype(np.intp)
        y0 = np.floor(ys).astype(np.intp)
        x1 = np.minimum(x0 + 1, w - 1)   # clamp: última linha/coluna preservadas
        y1 = np.minimum(y0 + 1, h - 1)
        wx, wy = xs - x0, ys - y0
        if img.ndim == 3:
            wx, wy = wx[..., None], wy[..., None]
        f = img.astype(np.float64)
        v = ((1 - wx) * (1 - wy) * f[y0, x0] + wx * (1 - wy) * f[y0, x1] +
             (1 - wx) * wy * f[y1, x0] + wx * wy * f[y1, x1])
        res = np.clip(np.floor(v + 0.5), 0, 255).astype(img.dtype)
    res[~dentro] = 0
    return res


def _validar_interp(interpolacao):
    if interpolacao not in ("vizinho", "bilinear"):
        raise ValueError("Interpolação deve ser 'vizinho' ou 'bilinear'.")


def escalar(img, sx, sy=None, interpolacao="vizinho", max_pixels=16_000_000):
    """Escala (sx, sy independentes) com mapeamento inverso. A saída TEM
    dimensões round(w*sx) x round(h*sy) (ampliação ou redução).

    Mapeamento inverso alinhado pelos cantos (base 0): x_src = x_dst*(w-1)/(W-1),
    de modo que primeira e última linha/coluna da origem são preservadas.
    Funciona em cinza (2D) e colorida (3D, canal a canal)."""
    img = np.asarray(img)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D ou 3D não vazia.")
    sy = sx if sy is None else sy
    sx, sy = float(sx), float(sy)
    if not np.isfinite([sx, sy]).all() or sx <= 0 or sy <= 0:
        raise ValueError("Os fatores de escala devem ser números positivos.")
    _validar_interp(interpolacao)
    h, w = img.shape[:2]
    W, H = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    if W * H > max_pixels:
        raise ValueError(f"A saída de {W}×{H} pixels excede o limite de {max_pixels:,}.")
    xd = np.arange(W, dtype=np.float64)
    yd = np.arange(H, dtype=np.float64)
    xs = xd * (w - 1) / (W - 1) if W > 1 else np.full(1, (w - 1) / 2)
    ys = yd * (h - 1) / (H - 1) if H > 1 else np.full(1, (h - 1) / 2)
    x_src, y_src = np.meshgrid(xs, ys)
    return _amostrar(img, x_src, y_src, interpolacao)


def rotacionar(img, angulo, pivo=None, interpolacao="vizinho", expandir=True,
               max_pixels=16_000_000):
    """Rotaciona em torno do pivô (x, y), em coordenadas base 0 da imagem
    (padrão: centro). Ângulo em graus, POSITIVO = SENTIDO HORÁRIO na tela
    (eixo y para baixo). Passos: translada pivô à origem -> gira -> translada
    de volta. Com expandir=True o canvas cresce para não cortar a imagem
    (round do bounding box dos 4 cantos); senão mantém o tamanho original."""
    img = np.asarray(img)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D ou 3D não vazia.")
    angulo = float(angulo)
    if not np.isfinite(angulo):
        raise ValueError("O ângulo deve ser finito.")
    _validar_interp(interpolacao)
    h, w = img.shape[:2]
    px, py = ((w - 1) / 2, (h - 1) / 2) if pivo is None else map(float, pivo)
    if not np.isfinite([px, py]).all():
        raise ValueError("O pivô deve conter coordenadas finitas.")
    t = np.deg2rad(angulo)
    c, s = np.cos(t), np.sin(t)
    if expandir:
        cx = np.array([0, w - 1, 0, w - 1], float) - px
        cy = np.array([0, 0, h - 1, h - 1], float) - py
        dx = px + c * cx - s * cy
        dy = py + s * cx + c * cy
        x0, y0 = np.floor(dx.min() + 1e-9), np.floor(dy.min() + 1e-9)
        W = int(np.ceil(dx.max() - 1e-9) - x0 + 1)
        H = int(np.ceil(dy.max() - 1e-9) - y0 + 1)
    else:
        x0 = y0 = 0
        W, H = w, h
    if W * H > max_pixels:
        raise ValueError(f"A saída de {W}×{H} pixels excede o limite de {max_pixels:,}.")
    yy, xx = np.indices((H, W), dtype=np.float64)
    x = xx + x0 - px
    y = yy + y0 - py
    # rotação inversa (R(-theta)) para o mapeamento inverso
    x_src = c * x + s * y + px
    y_src = -s * x + c * y + py
    return _amostrar(img, x_src, y_src, interpolacao)
