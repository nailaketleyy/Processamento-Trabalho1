import unittest

import numpy as np

from algoritmos import frequencia as fq
from algoritmos.cor import (
    ajustar_cmy, ajustar_hsi, chroma_key, componente_hsi, equalizar_hsi,
    hsi_para_rgb, negativo, rgb_para_hsi, escala_cinza, sepia,
)


class TestFourier(unittest.TestCase):
    def test_dft_direta_bate_fft_e_idft_reconstroi(self):
        rng = np.random.default_rng(20)
        dados = rng.random((5, 7))
        f = fq.dft2(dados)
        np.testing.assert_allclose(f, np.fft.fft2(dados), atol=1e-10)
        np.testing.assert_allclose(fq.idft2(f), dados, atol=1e-10)

    def test_limite_dft_direta(self):
        with self.assertRaises(ValueError):
            fq.dft2(np.zeros((65, 64)))

    def test_filtros_ideais_basicos_e_notch(self):
        forma = (21, 21)
        baixa = fq.filtro_ideal(forma, "passa-baixa", corte=2)
        alta = fq.filtro_ideal(forma, "rejeita-baixa", corte=2)
        self.assertEqual(baixa[10, 10], 1)
        self.assertEqual(alta[10, 10], 0)
        faixa = fq.filtro_ideal(
            forma, "passa-faixa", interno=3, externo=6)
        self.assertEqual(faixa[10, 14], 1)
        self.assertEqual(faixa[10, 10], 0)
        notch = fq.filtro_ideal(
            forma, "rejeita-notch", centros=[(0, 4)], raio_notch=1)
        self.assertEqual(notch[10, 14], 0)
        self.assertEqual(notch[10, 6], 0)

    def test_edicao_de_espectro_e_reconstrucao(self):
        dados = np.arange(64, dtype=np.uint8).reshape(8, 8)
        f = fq.transformar(dados)
        identidade = np.ones(dados.shape)
        np.testing.assert_array_equal(fq.reconstruir(fq.filtrar_espectro(f, identidade)), dados)
        mascara = fq.filtro_ideal(dados.shape, "passa-baixa", corte=0)
        saida = fq.reconstruir(fq.filtrar_espectro(f, mascara), saida="normalizar")
        self.assertEqual(saida.shape, dados.shape)

    def test_teorema_convolucao_kernel_identidade(self):
        dados = np.arange(30, dtype=np.float64).reshape(5, 6)
        kernel = np.zeros((3, 3))
        kernel[1, 1] = 1
        np.testing.assert_allclose(
            fq.convolucao_circular_frequencia(dados, kernel), dados, atol=1e-10)

    def test_teorema_convolucao_kernel_media_com_wraparound(self):
        dados = np.arange(30, dtype=np.float64).reshape(5, 6)
        kernel = np.ones((3, 3)) / 9
        esperado = np.zeros_like(dados)
        for y in range(3):
            for x in range(3):
                esperado += np.roll(
                    dados, (y - 1, x - 1), axis=(0, 1)) * kernel[y, x]
        np.testing.assert_allclose(
            fq.convolucao_circular_frequencia(dados, kernel), esperado, atol=1e-10)

    def test_aliasing_e_moire_preservam_formato_rgb(self):
        img = np.arange(8 * 10 * 3, dtype=np.uint8).reshape(8, 10, 3)
        self.assertEqual(fq.demo_aliasing(img, 2).shape, img.shape)
        self.assertEqual(fq.demo_aliasing(img, 2, suavizar=True).shape, img.shape)
        self.assertEqual(fq.padrao_moire(30, 20).shape, (20, 30))


class TestOperacoesDeCor(unittest.TestCase):
    def setUp(self):
        self.img = np.array(
            [[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
             [[100, 120, 140], [255, 255, 255], [0, 0, 0]]],
            dtype=np.uint8)

    def test_rgb_hsi_rgb_roundtrip(self):
        roundtrip = hsi_para_rgb(*rgb_para_hsi(self.img))
        np.testing.assert_allclose(roundtrip, self.img, atol=1)

    def test_equalizacao_ajustes_negativo_cinza_e_sepia(self):
        self.assertEqual(equalizar_hsi(self.img).shape, self.img.shape)
        self.assertEqual(componente_hsi(self.img, "H").shape, self.img.shape)
        self.assertEqual(componente_hsi(self.img, "S").shape, self.img.shape)
        self.assertEqual(componente_hsi(self.img, "I").shape, self.img.shape)
        np.testing.assert_array_equal(negativo(self.img)[0, 0], [0, 255, 255])
        self.assertEqual(ajustar_hsi(self.img, 15, 0.8, 1.1).shape, self.img.shape)
        self.assertEqual(ajustar_cmy(self.img, 10, -10, 20).shape, self.img.shape)
        media = escala_cinza(self.img, "media")
        ponderada = escala_cinza(self.img, "ponderada")
        self.assertEqual(media.shape, self.img.shape)
        self.assertEqual(ponderada.shape, self.img.shape)
        self.assertEqual(sepia(self.img).shape, self.img.shape)

    def test_chroma_key_substitui_verde_por_fundo(self):
        frente = np.array([[[0, 255, 0], [255, 0, 0]]], dtype=np.uint8)
        fundo = np.array([[[10, 20, 30]]], dtype=np.uint8)
        saida = chroma_key(frente, fundo, distancia=0)
        np.testing.assert_array_equal(saida, [[[10, 20, 30], [255, 0, 0]]])


if __name__ == "__main__":
    unittest.main()
