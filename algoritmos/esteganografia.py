"""ETAPA 1 - Esteganografia por bits menos significativos (LSB).

Cabeçalho (7 bytes = 56 bits, gravado com 1 bit por canal nos 56 primeiros valores):
    "ST" (2 bytes) | bits por canal k (1 byte) | tamanho do dado em bytes (4 bytes)
Em seguida o dado é gravado usando os k bits menos significativos de cada canal.
"""

import numpy as np

MAGIC = b"ST"
HEADER_BITS = 56


def capacidade(img, k):
    """Capacidade em bytes para k bits por canal."""
    return max(0, (img.size - HEADER_BITS) * k // 8)


def ocultar(img, dados: bytes, k):
    flat = img.reshape(-1).copy()
    if len(dados) > capacidade(img, k):
        raise ValueError("Dado grande demais para esta imagem. "
                         "Aumente os bits por canal ou use uma imagem maior.")
    cab = MAGIC + bytes([k]) + len(dados).to_bytes(4, "big")
    cab_bits = np.unpackbits(np.frombuffer(cab, dtype=np.uint8))
    flat[:HEADER_BITS] = (flat[:HEADER_BITS] & 0xFE) | cab_bits

    if len(dados) > 0:
        bits = np.unpackbits(np.frombuffer(dados, dtype=np.uint8))
        sobra = (-len(bits)) % k
        if sobra:
            bits = np.concatenate([bits, np.zeros(sobra, dtype=np.uint8)])
        pesos = (1 << np.arange(k - 1, -1, -1)).astype(np.uint16)
        valores = (bits.reshape(-1, k).astype(np.uint16) * pesos).sum(axis=1).astype(np.uint8)
        n = len(valores)
        mascara = (0xFF << k) & 0xFF
        trecho = flat[HEADER_BITS:HEADER_BITS + n]
        flat[HEADER_BITS:HEADER_BITS + n] = (trecho & mascara) | valores
    return flat.reshape(img.shape)


def extrair(img) -> bytes:
    flat = img.reshape(-1)
    if flat.size < HEADER_BITS:
        raise ValueError("Imagem pequena demais.")
    cab = np.packbits(flat[:HEADER_BITS] & 1).tobytes()
    if cab[:2] != MAGIC:
        raise ValueError("Nenhuma mensagem encontrada nesta imagem.\n"
                         "(Lembre-se: JPG destrói a mensagem; use BMP ou PNG.)")
    k = cab[2]
    tamanho = int.from_bytes(cab[3:7], "big")
    if not (1 <= k <= 8) or tamanho * 8 > (flat.size - HEADER_BITS) * k:
        raise ValueError("Cabeçalho inválido: a imagem provavelmente foi alterada.")
    if tamanho == 0:
        return b""
    n = -(-tamanho * 8 // k)
    valores = flat[HEADER_BITS:HEADER_BITS + n] & ((1 << k) - 1)
    bits = ((valores[:, None] >> np.arange(k - 1, -1, -1)) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bits[:tamanho * 8]).tobytes()
