import unittest

import numpy as np

from algoritmos.bordas import gradiente, sobel
from algoritmos.geometria import transformar


class TestDeteccaoDeBordas(unittest.TestCase):
    def setUp(self):
        cinza = np.zeros((12, 12), dtype=np.uint8)
        cinza[:, 6:] = 255
        self.img = np.repeat(cinza[:, :, None], 3, axis=2)

    def test_sobel_direcional_normalizado_e_rgb(self):
        sx = sobel(self.img, "x")
        sy = sobel(self.img, "y")
        self.assertEqual(sx.shape, self.img.shape)
        self.assertEqual(sx.dtype, np.uint8)
        np.testing.assert_array_equal(sx[..., 0], sx[..., 1])
        np.testing.assert_array_equal(sx[..., 1], sx[..., 2])
        self.assertGreater(int(sx.max()), 0)
        self.assertEqual(int(sy.max()), 0)

    def test_magnitude_gradiente_desenha_borda_em_cinza(self):
        resultado = gradiente(self.img)
        self.assertEqual(resultado.shape, self.img.shape)
        self.assertEqual(resultado.dtype, np.uint8)
        np.testing.assert_array_equal(resultado[..., 0], resultado[..., 1])
        np.testing.assert_array_equal(resultado[..., 1], resultado[..., 2])
        self.assertGreater(int(resultado.max()), 0)

    def test_transformacao_identidade_em_ambos_metodos(self):
        for metodo in ("vizinho", "bilinear"):
            resultado = transformar(self.img, interpolacao=metodo)
            np.testing.assert_array_equal(resultado, self.img)

    def test_translacao_com_vizinho_preenche_fundo_e_mantem_formato(self):
        img = np.full((5, 6, 3), 120, dtype=np.uint8)
        resultado = transformar(
            img, deslocamento=(1, 0), pivo=(0, 0), interpolacao="vizinho")
        self.assertEqual(resultado.shape, img.shape)
        np.testing.assert_array_equal(resultado[:, 0], 0)
        np.testing.assert_array_equal(resultado[:, 1:], img[:, :-1])

    def test_rotacao_de_quarto_de_volta_em_torno_do_pivo(self):
        img = np.zeros((5, 5), dtype=np.uint8)
        img[2, 3] = 255
        resultado = transformar(
            img, angulo=90, pivo=(2, 2), interpolacao="vizinho")
        self.assertEqual(int(resultado[3, 2]), 255)

    def test_interpolacao_bilinear_cria_valores_intermediarios(self):
        img = np.array(
            [[0, 60, 120], [90, 150, 210], [180, 240, 255]], dtype=np.uint8)
        resultado = transformar(
            img, escala=(1.5, 1.5), pivo=(1, 1),
            interpolacao="bilinear")
        self.assertEqual(int(resultado[1, 2]), 190)

    def test_rejeita_escala_nao_positiva(self):
        with self.assertRaises(ValueError):
            transformar(self.img, escala=(0, 1))


if __name__ == "__main__":
    unittest.main()
