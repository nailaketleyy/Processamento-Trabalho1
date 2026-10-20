"""Transformada de Fourier 2D, filtros ideais e demonstrações de aliasing."""

import numpy as np


def dft2(dados):
    """Transformada discreta 2D direta, implementada sem FFT externa.

    A implementação separável é O(M²N + MN²); adequada para estudo
    e imagens pequenas.
    """
    dados = np.asarray(dados, dtype=np.float64)
    if dados.ndim != 2 or dados.size == 0:
        raise ValueError("A DFT requer uma matriz 2D não vazia.")
    m, n = dados.shape
    if m * n > 4096:
        raise ValueError("A DFT direta está limitada a 4096 amostras; use FFT NumPy.")
    y = np.arange(m)
    x = np.arange(n)
    wm = np.exp(-2j * np.pi * np.outer(y, y) / m)
    wn = np.exp(-2j * np.pi * np.outer(x, x) / n)
    return wm @ dados @ wn.T


def idft2(espectro):
    """Transformada inversa 2D direta, implementada sem FFT externa."""
    espectro = np.asarray(espectro, dtype=np.complex128)
    if espectro.ndim != 2 or espectro.size == 0:
        raise ValueError("A IDFT requer uma matriz 2D não vazia.")
    m, n = espectro.shape
    if m * n > 4096:
        raise ValueError("A IDFT direta está limitada a 4096 amostras; use FFT NumPy.")
    y = np.arange(m)
    x = np.arange(n)
    wm = np.exp(2j * np.pi * np.outer(y, y) / m)
    wn = np.exp(2j * np.pi * np.outer(x, x) / n)
    return wm @ espectro @ wn.T / (m * n)


def _fft_radix2_1d(dados, inversa=False):
    valores = np.asarray(dados, dtype=np.complex128).copy()
    tamanho = valores.size
    if tamanho == 0 or tamanho & (tamanho - 1):
        raise ValueError("A FFT própria requer dimensões que sejam potências de 2.")

    indice_invertido = 0
    for indice in range(1, tamanho):
        bit = tamanho >> 1
        while indice_invertido & bit:
            indice_invertido ^= bit
            bit >>= 1
        indice_invertido ^= bit
        if indice < indice_invertido:
            valores[indice], valores[indice_invertido] = (
                valores[indice_invertido], valores[indice])

    sinal = 1 if inversa else -1
    largura = 2
    while largura <= tamanho:
        metade = largura // 2
        fatores = np.exp(sinal * 2j * np.pi * np.arange(metade) / largura)
        for inicio in range(0, tamanho, largura):
            par = valores[inicio:inicio + metade].copy()
            impar = valores[inicio + metade:inicio + largura].copy() * fatores
            valores[inicio:inicio + metade] = par + impar
            valores[inicio + metade:inicio + largura] = par - impar
        largura *= 2
    if inversa:
        valores /= tamanho
    return valores


def _fft2_radix2(dados, inversa=False):
    dados = np.asarray(dados, dtype=np.complex128)
    if dados.ndim != 2 or dados.size == 0:
        raise ValueError("A FFT própria requer uma matriz 2D não vazia.")
    if any(tamanho & (tamanho - 1) for tamanho in dados.shape):
        raise ValueError(
            "A FFT própria radix-2 requer altura e largura potências de 2; "
            "use FFT NumPy para outras dimensões.")
    linhas = np.stack([_fft_radix2_1d(linha, inversa) for linha in dados])
    colunas = np.stack([
        _fft_radix2_1d(linhas[:, coluna], inversa)
        for coluna in range(linhas.shape[1])
    ], axis=1)
    return colunas


def transformar(dados, metodo="fft"):
    if metodo == "dft":
        return dft2(dados)
    if metodo == "fft":
        return np.fft.fft2(np.asarray(dados, dtype=np.float64))
    if metodo == "fft-propria":
        return _fft2_radix2(dados)
    raise ValueError("Método deve ser 'dft' ou 'fft'.")


def inversa(espectro, metodo="fft"):
    if metodo == "dft":
        return idft2(espectro)
    if metodo == "fft":
        return np.fft.ifft2(np.asarray(espectro, dtype=np.complex128))
    if metodo == "fft-propria":
        return _fft2_radix2(espectro, inversa=True)
    raise ValueError("Método deve ser 'dft' ou 'fft'.")


def espectro_exibicao(espectro):
    """Retorna log(1+|F|) centralizado, em uint8, para exibição."""
    magnitude = np.log1p(np.abs(np.fft.fftshift(espectro)))
    lo, hi = magnitude.min(), magnitude.max()
    if hi <= lo:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.floor((magnitude - lo) * 255 / (hi - lo) + 0.5).astype(np.uint8)


def _distancias_centro(shape):
    m, n = shape
    y, x = np.indices(shape, dtype=np.float64)
    return np.hypot(y - m // 2, x - n // 2)


def filtro_ideal(shape, tipo, corte=20.0, interno=10.0, externo=30.0,
                 centros=(), raio_notch=5.0):
    """Cria máscara ideal passa/rejeita baixa, alta, faixa ou notch.

    `centros` contém centros notch em coordenadas relativas ao centro,
    ordenadas como (u, v). Cada notch é aplicado junto ao seu simétrico.
    """
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError("A forma deve conter altura e largura positivas.")
    valores = (float(corte), float(interno), float(externo), float(raio_notch))
    if not np.isfinite(valores).all() or min(valores) < 0:
        raise ValueError("Os cortes e raios devem ser finitos e não negativos.")
    tipo = tipo.lower()
    d = _distancias_centro(shape)
    baixa = d <= corte
    if tipo == "passa-baixa":
        return baixa.astype(np.float64)
    if tipo == "rejeita-baixa":
        return (~baixa).astype(np.float64)
    if tipo in ("passa-faixa", "rejeita-faixa"):
        if interno > externo:
            raise ValueError("O corte interno deve ser menor ou igual ao externo.")
        faixa = (d >= interno) & (d <= externo)
        return faixa.astype(np.float64) if tipo == "passa-faixa" else (~faixa).astype(np.float64)
    if tipo in ("passa-notch", "rejeita-notch"):
        y, x = np.indices(shape, dtype=np.float64)
        cy, cx = shape[0] // 2, shape[1] // 2
        notch = np.zeros(shape, dtype=bool)
        for u, v in centros:
            u, v = float(u), float(v)
            if not np.isfinite([u, v]).all():
                raise ValueError("Os centros notch devem ser finitos.")
            for su, sv in ((u, v), (-u, -v)):
                dy = (y - cy - su + shape[0] / 2) % shape[0] - shape[0] / 2
                dx = (x - cx - sv + shape[1] / 2) % shape[1] - shape[1] / 2
                notch |= dy ** 2 + dx ** 2 <= raio_notch ** 2
        return notch.astype(np.float64) if tipo == "passa-notch" else (~notch).astype(np.float64)
    raise ValueError(f"Tipo de filtro ideal desconhecido: {tipo!r}.")


def filtrar_espectro(espectro, mascara):
    """Multiplica o espectro não deslocado por uma máscara centrada."""
    f = np.asarray(espectro, dtype=np.complex128)
    h = np.asarray(mascara, dtype=np.float64)
    if f.ndim != 2 or h.shape != f.shape:
        raise ValueError("A máscara e o espectro devem ter a mesma forma 2D.")
    if not np.isfinite(h).all():
        raise ValueError("A máscara deve conter valores finitos.")
    return f * np.fft.ifftshift(np.clip(h, 0.0, 1.0))


def reconstruir(espectro, metodo="fft", saida="saturar"):
    """Inverte o espectro e retorna intensidades uint8 para a imagem."""
    imagem = np.real(inversa(espectro, metodo))
    if saida == "normalizar":
        minimo, maximo = imagem.min(), imagem.max()
        if maximo > minimo:
            imagem = (imagem - minimo) * (255.0 / (maximo - minimo))
        else:
            imagem = np.zeros_like(imagem)
    elif saida != "saturar":
        raise ValueError("Saída deve ser 'saturar' ou 'normalizar'.")
    return np.clip(np.floor(imagem + 0.5), 0, 255).astype(np.uint8)


def filtrar_rgb(img, mascara, metodo="fft", saida="saturar"):
    """Filtra os três canais RGB com a mesma máscara de frequência."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        raise ValueError("A imagem deve ser RGB e não vazia.")
    resultado = np.empty(img.shape, dtype=np.float64)
    for canal in range(3):
        espectro = transformar(img[..., canal], metodo)
        filtrado = filtrar_espectro(espectro, mascara)
        resultado[..., canal] = np.real(inversa(filtrado, metodo))
    if saida == "normalizar":
        minimo, maximo = resultado.min(), resultado.max()
        if maximo > minimo:
            resultado = (resultado - minimo) * (255.0 / (maximo - minimo))
        else:
            resultado.fill(0)
    elif saida != "saturar":
        raise ValueError("Saída deve ser 'saturar' ou 'normalizar'.")
    return np.clip(np.floor(resultado + 0.5), 0, 255).astype(np.uint8)


def convolucao_circular_frequencia(img, kernel, metodo="fft"):
    """Convolução circular 2D usando o teorema da convolução.

    O centro do kernel é deslocado para (0,0), para que o resultado fique
    alinhado com a definição espacial da máscara.
    """
    img = np.asarray(img, dtype=np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)
    if img.ndim != 2 or img.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D não vazia.")
    if kernel.ndim != 2 or not np.isfinite(kernel).all():
        raise ValueError("O kernel deve ser uma matriz 2D finita.")
    if kernel.shape[0] > img.shape[0] or kernel.shape[1] > img.shape[1]:
        raise ValueError("O kernel não pode ser maior do que a imagem.")
    h, w = img.shape
    kp = np.zeros((h, w), dtype=np.float64)
    kh, kw = kernel.shape
    kp[:kh, :kw] = kernel
    kp = np.roll(kp, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return np.real(inversa(transformar(img, metodo) * transformar(kp, metodo), metodo))


def convolucao_circular_rgb(img, kernel, metodo="fft"):
    """Aplica a convolução circular independentemente aos canais RGB."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        raise ValueError("A imagem deve ser RGB e não vazia.")
    canais = [convolucao_circular_frequencia(img[..., c], kernel, metodo)
              for c in range(3)]
    return np.stack(canais, axis=-1)


def demo_aliasing(img, fator=2, suavizar=False):
    """Subamostra e amplia por vizinho mais próximo para mostrar aliasing.

    Com suavizar=True, aplica antes um passa-baixa gaussiano simples.
    """
    img = np.asarray(img)
    if img.ndim not in (2, 3) or img.size == 0:
        raise ValueError("A imagem deve ser uma matriz 2D ou 3D não vazia.")
    if not isinstance(fator, (int, np.integer)) or fator < 2:
        raise ValueError("O fator de subamostragem deve ser inteiro >= 2.")
    if suavizar:
        from .convolucao import correlacao, kernel_gaussiano
        dados = img.astype(np.float64)
        k = kernel_gaussiano(5, max(0.8, fator / 2))
        filtrada = correlacao(dados, k, "replicar")
        img = np.clip(np.floor(filtrada + 0.5), 0, 255).astype(np.uint8)
    h, w = img.shape[:2]
    amostrada = img[::fator, ::fator]
    ys = np.minimum(np.arange(h) // fator, amostrada.shape[0] - 1)
    xs = np.minimum(np.arange(w) // fator, amostrada.shape[1] - 1)
    return amostrada[ys][:, xs]


def padrao_moire(largura=256, altura=256, frequencia=0.12, diferenca=0.006):
    """Gera um padrão de franjas por superposição de duas grades sinusoidais."""
    if largura < 1 or altura < 1:
        raise ValueError("A largura e altura devem ser positivas.")
    if not np.isfinite([frequencia, diferenca]).all() or frequencia <= 0 or diferenca < 0:
        raise ValueError("Frequência deve ser positiva e diferença não negativa.")
    _, x = np.indices((altura, largura), dtype=np.float64)
    a = np.cos(2 * np.pi * frequencia * x)
    b = np.cos(2 * np.pi * (frequencia + diferenca) * x)
    return np.floor((a + b + 2) * 255 / 4 + 0.5).astype(np.uint8)
