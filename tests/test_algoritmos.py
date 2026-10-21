import unittest
from unittest.mock import patch

import numpy as np

from algoritmos import convolucao as cv
from algoritmos.histograma import equalizar_cinza, histograma
from algoritmos.intensidade import limiarizar, lut_negativo


class TestOperacoesEspaciais(unittest.TestCase):
    def test_convolucao_preserva_formato_e_identidade(self):
        img = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        resultado = cv.filtrar(
            img, cv.kernel_identidade(3), borda="replicar", saida="saturar")
        np.testing.assert_array_equal(resultado, img)
        self.assertEqual(resultado.shape, img.shape)

    def test_media_tem_soma_unitaria_apos_normalizacao(self):
        img = np.full((7, 7), 90, dtype=np.uint8)
        kernel = cv.kernel_media(5)
        kernel /= kernel.sum()
        resultado = cv.filtrar(img, kernel, borda="replicar")
        np.testing.assert_array_equal(resultado, img)

    def test_mediana_remove_ruido_impulsivo_e_preserva_dimensoes(self):
        img = np.full((7, 9), 100, dtype=np.uint8)
        img[3, 4] = 255
        resultado = cv.mediana(img, 3, borda="replicar")
        self.assertEqual(resultado.shape, img.shape)
        self.assertEqual(int(resultado[3, 4]), 100)

    def test_mediana_nao_usa_numpy_median(self):
        img = np.arange(25, dtype=np.uint8).reshape(5, 5)
        with patch("algoritmos.convolucao.np.median", side_effect=AssertionError):
            resultado = cv.mediana(img, 3)
        self.assertEqual(resultado.shape, img.shape)

    def test_kernels_de_aguçamento(self):
        laplaciano = cv.kernel_aguçamento_laplaciano(5)
        high_boost = cv.kernel_high_boost(5, A=1.5, sigma=1.0)
        self.assertAlmostEqual(float(laplaciano.sum()), 1.0)
        self.assertAlmostEqual(float(high_boost.sum()), 0.5)
        self.assertEqual(laplaciano.shape, (5, 5))
        self.assertEqual(high_boost.shape, (5, 5))

    def test_rejeita_tamanho_par(self):
        with self.assertRaises(ValueError):
            cv.mediana(np.zeros((4, 4), dtype=np.uint8), 4)


class TestOperacoesDeIntensidadeEHistograma(unittest.TestCase):
    def test_histograma_e_equalizacao_constante(self):
        img = np.full((2, 3), 37, dtype=np.uint8)
        self.assertEqual(int(histograma(img)[37]), img.size)
        np.testing.assert_array_equal(equalizar_cinza(img), np.full_like(img, 255))

    def test_negativo_e_limiarizacao(self):
        np.testing.assert_array_equal(lut_negativo()[[0, 255]], [255, 0])
        img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        resultado = limiarizar(img, 128)
        np.testing.assert_array_equal(
            resultado, np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
