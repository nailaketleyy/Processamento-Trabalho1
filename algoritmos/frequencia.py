"""Transformada de Fourier 2D, filtros ideais e demonstrações de aliasing."""

import numpy as np

LIMITE_DFT = 1024  # maior dimensão aceita pela DFT direta O(N²) (matriz W de N×N)


def dft2(dados):
    """Transformada discreta 2D direta, implementada sem FFT externa.

    A implementação separável é O(M²N + MN²); adequada para estudo
    e imagens pequenas.
    """
    dados = np.asarray(dados, dtype=np.float64)
    if dados.ndim != 2 or dados.size == 0:
        raise ValueError("A DFT requer uma matriz 2D não vazia.")
    m, n = dados.shape
    if max(m, n) > LIMITE_DFT:
        raise ValueError(f"A DFT direta está limitada a {LIMITE_DFT} amostras por dimensão; "
                         "use a FFT própria ou a FFT de biblioteca.")
    y = np.arange(m)
    x = np.arange(n)
    wm = np.exp(-2j * np.pi * (np.outer(y, y) % m) / m)
    wn = np.exp(-2j * np.pi * (np.outer(x, x) % n) / n)
    return wm @ dados @ wn.T


def idft2(espectro):
    """Transformada inversa 2D direta, implementada sem FFT externa."""
    espectro = np.asarray(espectro, dtype=np.complex128)
    if espectro.ndim != 2 or espectro.size == 0:
        raise ValueError("A IDFT requer uma matriz 2D não vazia.")
    m, n = espectro.shape
    if max(m, n) > LIMITE_DFT:
        raise ValueError(f"A IDFT direta está limitada a {LIMITE_DFT} amostras por dimensão; "
                         "use a FFT própria ou a FFT de biblioteca.")
    y = np.arange(m)
    x = np.arange(n)
    wm = np.exp(2j * np.pi * (np.outer(y, y) % m) / m)
    wn = np.exp(2j * np.pi * (np.outer(x, x) % n) / n)
    return wm @ espectro @ wn.T / (m * n)


def _fft_radix2_eixo(dados, eixo, inversa=False):
    """FFT radix-2 (Cooley-Tukey, duplicação sucessiva) ao longo de `eixo`,
    aplicada a todas as linhas/colunas de uma vez (vetorizada)."""
    v = np.moveaxis(np.asarray(dados, dtype=np.complex128), eixo, -1).copy()
    n = v.shape[-1]
    if n == 0 or n & (n - 1):
        raise ValueError("A FFT própria requer dimensões que sejam potências de 2.")
    # permutação por inversão de bits
    bits = n.bit_length() - 1
    idx = np.arange(n)
    rev = np.zeros(n, dtype=np.intp)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    v = v[..., rev]
    sinal = 1 if inversa else -1
    tam = 2
    while tam <= n:
        meio = tam // 2
        w = np.exp(sinal * 2j * np.pi * np.arange(meio) / tam)
        blocos = v.reshape(v.shape[:-1] + (n // tam, tam))
        par = blocos[..., :meio].copy()
        impar = blocos[..., meio:] * w
        blocos[..., :meio] = par + impar
        blocos[..., meio:] = par - impar
        v = blocos.reshape(v.shape)
        tam *= 2
    if inversa:
        v /= n
    return np.moveaxis(v, -1, eixo)


def _fft2_radix2(dados, inversa=False):
    dados = np.asarray(dados, dtype=np.complex128)
    if dados.ndim != 2 or dados.size == 0:
        raise ValueError("A FFT própria requer uma matriz 2D não vazia.")
    if any(t & (t - 1) for t in dados.shape):
        raise ValueError(
            "A FFT própria radix-2 requer altura e largura potências de 2; "
            "use o preenchimento (zero padding) ou a FFT de biblioteca.")
    return _fft_radix2_eixo(_fft_radix2_eixo(dados, 1, inversa), 0, inversa)


def transformar(dados, metodo="fft"):
    if metodo == "dft":
        return dft2(dados)
    if metodo == "fft":
        return np.fft.fft2(np.asarray(dados, dtype=np.float64))
    if metodo == "fft-propria":
        return _fft2_radix2(dados)
    raise ValueError("Método deve ser 'dft', 'fft' ou 'fft-propria'.")


def inversa(espectro, metodo="fft"):
    if metodo == "dft":
        return idft2(espectro)
    if metodo == "fft":
        return np.fft.ifft2(np.asarray(espectro, dtype=np.complex128))
    if metodo == "fft-propria":
        return _fft2_radix2(espectro, inversa=True)
    raise ValueError("Método deve ser 'dft', 'fft' ou 'fft-propria'.")


def deslocar(a):
    """fftshift próprio: leva a frequência zero (0,0) para o centro (M//2, N//2)."""
    a = np.asarray(a)
    return np.roll(a, (a.shape[0] // 2, a.shape[1] // 2), axis=(0, 1))


def desfazer_deslocamento(a):
    """ifftshift próprio (inverso de `deslocar`, vale também p/ dimensões ímpares)."""
    a = np.asarray(a)
    return np.roll(a, (-(a.shape[0] // 2), -(a.shape[1] // 2)), axis=(0, 1))


def centralizar_entrada(dados):
    """Alternativa clássica ao shift: multiplica f(x,y) por (-1)^(x+y) ANTES da DFT;
    o espectro já sai com a origem no centro (dimensões pares)."""
    dados = np.asarray(dados, dtype=np.float64)
    y, x = np.indices(dados.shape)
    return dados * np.where((x + y) % 2 == 0, 1.0, -1.0)


def forma_preenchida(shape, metodo="fft"):
    """Dimensões P×Q do preenchimento com zeros: P >= 2M-1 e Q >= 2N-1
    (aqui P=2M, Q=2N; potência de 2 quando o método é a FFT própria)."""
    m, n = shape[:2]
    p, q = 2 * m, 2 * n
    if metodo == "fft-propria":
        p, q = 1 << (p - 1).bit_length(), 1 << (q - 1).bit_length()
    return p, q


def preencher_zeros(canal, forma):
    """Coloca o canal no canto superior esquerdo de uma matriz P×Q de zeros."""
    saida = np.zeros(forma, dtype=np.float64)
    saida[:canal.shape[0], :canal.shape[1]] = canal
    return saida


def espectro_exibicao(espectro):
    """Retorna log(1+|F|) centralizado, em uint8, para exibição."""
    magnitude = np.log1p(np.abs(deslocar(espectro)))
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
    return f * desfazer_deslocamento(np.clip(h, 0.0, 1.0))


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
    """Filtra R, G, B com a mesma máscara H(u,v) (centrada, valores em [0,1]).

    Se a máscara tem a forma PxQ de `forma_preenchida(...)`, aplica o pipeline
    completo: zero padding -> DFT -> H*F -> IDFT -> parte real -> recorte MxN
    (sem erro de wraparound). Se tem a forma MxN da imagem, filtra sem
    preenchimento (convolução circular)."""
    img = np.asarray(img)
    if img.ndim != 3 or img.shape[2] != 3 or img.size == 0:
        raise ValueError("A imagem deve ser RGB e não vazia.")
    m, n = img.shape[:2]
    mascara = np.asarray(mascara)
    if mascara.shape == (m, n):
        forma = (m, n)
    elif mascara.shape == forma_preenchida((m, n), metodo):
        forma = mascara.shape
    else:
        raise ValueError(
            f"A máscara {mascara.shape} não corresponde à imagem {(m, n)} nem ao "
            f"tamanho preenchido {forma_preenchida((m, n), metodo)}.")
    resultado = np.empty(img.shape, dtype=np.float64)
    for canal in range(3):
        f = transformar(preencher_zeros(img[..., canal], forma), metodo)
        g = np.real(inversa(filtrar_espectro(f, mascara), metodo))
        resultado[..., canal] = g[:m, :n]
    if saida == "normalizar":
        minimo, maximo = resultado.min(), resultado.max()
        if maximo > minimo:
            resultado = (resultado - minimo) * (255.0 / (maximo - minimo))
        else:
            resultado.fill(0)
    elif saida != "saturar":
        raise ValueError("Saída deve ser 'saturar' ou 'normalizar'.")
    return np.clip(np.floor(resultado + 0.5), 0, 255).astype(np.uint8)


def filtro_suave(shape, familia, tipo, corte=20.0, largura=10.0, ordem=2):
    """Filtros Butterworth e Gaussiano (opcionais). familia: 'butterworth'|'gaussiano';
    tipo: 'passa-baixa'|'passa-alta'|'passa-faixa'|'rejeita-faixa'. Máscara centrada."""
    if familia not in ("butterworth", "gaussiano"):
        raise ValueError("Família deve ser 'butterworth' ou 'gaussiano'.")
    if corte <= 0 or largura <= 0 or ordem < 1:
        raise ValueError("Corte e largura devem ser > 0 e a ordem >= 1.")
    d = _distancias_centro(shape)
    if tipo in ("passa-baixa", "passa-alta"):
        if familia == "butterworth":
            baixa = 1.0 / (1.0 + (d / corte) ** (2 * ordem))
        else:
            baixa = np.exp(-(d ** 2) / (2.0 * corte ** 2))
        return baixa if tipo == "passa-baixa" else 1.0 - baixa
    if tipo in ("passa-faixa", "rejeita-faixa"):
        d2 = d ** 2
        com_zero = np.where(d2 == corte ** 2, 1e-12, d2 - corte ** 2)
        if familia == "butterworth":
            rejeita = 1.0 / (1.0 + ((d * largura) / com_zero) ** (2 * ordem))
        else:
            rejeita = 1.0 - np.exp(-((d2 - corte ** 2) / (d * largura + 1e-12)) ** 2)
        return rejeita if tipo == "rejeita-faixa" else 1.0 - rejeita
    raise ValueError(f"Tipo de filtro desconhecido: {tipo!r}.")


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
