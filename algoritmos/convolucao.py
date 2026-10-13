"""ETAPA 2 (Cap. 3, seções 3.4 e 3.5) - Filtragem espacial linear por convolução.

Correlação  (w ☆ f):  g(x,y) = soma_s soma_t w(s,t) f(x+s, y+t)
Convolução  (w ★ f):  g(x,y) = soma_s soma_t w(s,t) f(x-s, y-t)   (= correlação com w girado 180°)

O filtro w é uma matriz m x n (aqui, sempre quadrada de lado ímpar).
A imagem de saída tem o MESMO tamanho da original: a borda é tratada com
preenchimento (padding), conforme a opção escolhida.

Implementação: para cada coeficiente w(s,t) soma-se à saída a imagem deslocada
multiplicada pelo coeficiente (vetorizado com NumPy; cada pixel recebe exatamente
a soma de produtos da definição).
"""

import numpy as np

BORDAS = ("zeros", "replicar", "espelhar")
SAIDAS = ("saturar", "absoluto", "normalizar")


# ---------------------------------------------------------------- preenchimento
def _indices(n, p, modo):
    """Índices de 0..n-1 estendidos p posições para cada lado."""
    i = np.arange(-p, n + p)
    if modo == "replicar":                 # repete o valor da borda
        return np.clip(i, 0, n - 1)
    if n == 1:
        return np.zeros_like(i)
    periodo = 2 * (n - 1)                  # espelhar: ... 2 1 0 | 0 1 2 ... (reflexão)
    i = i % periodo
    return np.where(i >= n, periodo - i, i)


def preencher(img, a, b, modo="zeros"):
    """Acrescenta a linhas (acima e abaixo) e b colunas (à esquerda e à direita)."""
    h, w = img.shape[:2]
    if modo == "zeros":
        saida = np.zeros((h + 2 * a, w + 2 * b) + img.shape[2:], dtype=img.dtype)
        saida[a:a + h, b:b + w] = img
        return saida
    return img[_indices(h, a, modo)][:, _indices(w, b, modo)]


# ---------------------------------------------------------------- filtragem
def correlacao(img, w, borda="zeros"):
    """Correlação de img (2D ou 3D) com o filtro w (m x n, ímpares). Devolve float64."""
    m, n = w.shape
    a, b = m // 2, n // 2
    h, larg = img.shape[:2]
    f = preencher(img.astype(np.float64), a, b, borda)
    g = np.zeros(img.shape, dtype=np.float64)
    for s in range(m):
        for t in range(n):
            coef = w[s, t]
            if coef != 0:
                g += coef * f[s:s + h, t:t + larg]
    return g


def convolucao(img, w, borda="zeros"):
    """Convolução = correlação com o filtro girado 180°."""
    return correlacao(img, w[::-1, ::-1], borda)


def para_uint8(g, saida="saturar"):
    """Leva o resultado (float, pode sair de [0,255]) de volta para 8 bits.
    saturar:    valores < 0 viram 0 e > 255 viram 255
    absoluto:   módulo do valor, depois saturado (útil p/ filtros derivativos)
    normalizar: reescala linearmente [mínimo, máximo] para [0, 255]"""
    if saida == "absoluto":
        g = np.abs(g)
    elif saida == "normalizar":
        lo, hi = g.min(), g.max()
        g = (g - lo) * 255.0 / (hi - lo) if hi > lo else np.zeros_like(g)
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)


def filtrar(img, kernel, usar_convolucao=True, borda="zeros", saida="saturar"):
    """Aplica o filtro e devolve imagem uint8 com o mesmo tamanho da original."""
    k = np.asarray(kernel, dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2 == 0:
        raise ValueError("O filtro deve ser quadrado com dimensão ímpar (3x3, 5x5, ...).")
    g = convolucao(img, k, borda) if usar_convolucao else correlacao(img, k, borda)
    return para_uint8(g, saida)


# ---------------------------------------------------------------- filtros prontos
def kernel_identidade(n):
    k = np.zeros((n, n))
    k[n // 2, n // 2] = 1
    return k


def kernel_media(n):
    """Filtro da média (caixa): todos os coeficientes iguais (a soma é 1 depois de normalizar)."""
    return np.ones((n, n))


def kernel_gaussiano(n, sigma):
    a = n // 2
    eixo = np.arange(-a, a + 1)
    xx, yy = np.meshgrid(eixo, eixo)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return k / k.sum()


# filtros 3x3 clássicos (com "saída" sugerida)
KERNELS_3X3 = {
    "Laplaciano":                 ([[0, 1, 0], [1, -4, 1], [0, 1, 0]], "absoluto"),
    "Sobel (bordas verticais)":   ([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "absoluto"),
    "Sobel (bordas horizontais)": ([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], "absoluto"),
    "Média ponderada (1-2-1)":   ([[1, 2, 1], [2, 4, 2], [1, 2, 1]], "saturar"),   # Fig. 3.32(b)
    "Nitidez (realce)":           ([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], "saturar"),
}


def redimensionar_kernel(k, n):
    """Corta ou completa com zeros (mantendo o centro) para n x n."""
    saida = np.zeros((n, n))
    m = k.shape[0]
    c = min(m, n)
    ok, on = m // 2, n // 2
    for i in range(-(c // 2), c // 2 + 1):
        for j in range(-(c // 2), c // 2 + 1):
            saida[on + i, on + j] = k[ok + i, ok + j]
    return saida
