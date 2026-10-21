import unittest
import numpy as np

from algoritmos import bordas, convolucao as cv, cor, frequencia as fq
from algoritmos.geometria import escalar, rotacionar
from algoritmos.intensidade import validar_pontos, lut_contraste


class TestGeometria(unittest.TestCase):
    def setUp(self):
        self.img = np.random.default_rng(0).integers(0, 256, (10, 12, 3), dtype=np.uint8)

    def test_escala_altera_dimensoes(self):
        for m in ("vizinho", "bilinear"):
            self.assertEqual(escalar(self.img, 2, 2, m).shape, (20, 24, 3))
            self.assertEqual(escalar(self.img, 0.5, 0.5, m).shape, (5, 6, 3))
            self.assertEqual(escalar(self.img[..., 0], 1.5, 1, m).shape, (10, 18))

    def test_rotacao_90_horario_e_expansao(self):
        for m in ("vizinho", "bilinear"):
            np.testing.assert_array_equal(rotacionar(self.img, 90, None, m),
                                          np.rot90(self.img, -1))
            np.testing.assert_array_equal(rotacionar(self.img, 0, None, m), self.img)
        self.assertGreater(rotacionar(self.img, 45).shape[0], 10)
        self.assertEqual(rotacionar(self.img, 45, expandir=False).shape, self.img.shape)

    def test_pivo_fora_do_centro(self):
        r = rotacionar(self.img, 0, pivo=(0, 0))
        np.testing.assert_array_equal(r, self.img)


class TestNitidezBordasCor(unittest.TestCase):
    def setUp(self):
        self.img = np.random.default_rng(1).integers(0, 256, (16, 16, 3), dtype=np.uint8)

    def test_laplacianos(self):
        self.assertEqual(cv.kernel_laplaciano(4)[1, 1], -4)
        self.assertEqual(cv.kernel_laplaciano(8)[1, 1], -8)
        self.assertEqual(cv.kernel_laplaciano(8, False)[1, 1], 8)
        self.assertEqual(cv.kernel_laplaciano(8).sum(), 0)
        plano = np.full((8, 8, 3), 90, dtype=np.uint8)
        np.testing.assert_array_equal(cv.agucar_laplaciano(plano, 8), plano)

    def test_unsharp_k0_identidade(self):
        np.testing.assert_array_equal(cv.unsharp_highboost(self.img, 0.0), self.img)

    def test_gradiente_aproximacao(self):
        a = bordas.gradiente(self.img)
        b = bordas.gradiente(self.img, aproximacao=True)
        self.assertEqual(a.shape, b.shape)

    def test_chroma_key_cor_escolhida(self):
        frente = np.zeros((4, 4, 3), dtype=np.uint8)
        frente[:2] = (255, 0, 0)
        fundo = np.full((4, 4, 3), 7, dtype=np.uint8)
        for esp in ("rgb", "hsi"):
            r = cor.chroma_key(frente, fundo, 30, (255, 0, 0), esp)
            self.assertTrue((r[:2] == 7).all() and (r[2:] == 0).all())

    def test_validacao_contraste(self):
        with self.assertRaises(ValueError):
            validar_pontos(100, 0, 50, 255)
        with self.assertRaises(ValueError):
            validar_pontos(10, 200, 50, 100)
        self.assertEqual(lut_contraste(1.0)[0], 0)


class TestFrequenciaPadding(unittest.TestCase):
    def test_padding_e_filtro_identidade(self):
        img = np.random.default_rng(2).integers(0, 256, (20, 30, 3), dtype=np.uint8)
        for m in ("fft", "fft-propria", "dft"):
            forma = fq.forma_preenchida(img.shape, m)
            self.assertGreaterEqual(forma[0], 2 * 20 - 1)
            self.assertGreaterEqual(forma[1], 2 * 30 - 1)
            np.testing.assert_array_equal(fq.filtrar_rgb(img, np.ones(forma), m), img)

    def test_shift_proprio_e_fft_vetorizada(self):
        a = np.random.default_rng(3).random((8, 16))
        np.testing.assert_allclose(fq.deslocar(a), np.fft.fftshift(a))
        np.testing.assert_allclose(fq.transformar(a, "fft-propria"), np.fft.fft2(a), atol=1e-10)
        c = fq.transformar(fq.centralizar_entrada(a), "fft")
        np.testing.assert_allclose(c, fq.deslocar(np.fft.fft2(a)), atol=1e-9)

    def test_filtros_suaves(self):
        h = fq.filtro_suave((32, 32), "butterworth", "passa-baixa", 8, ordem=2)
        self.assertAlmostEqual(h[16, 16], 1.0)
        g = fq.filtro_suave((32, 32), "gaussiano", "passa-alta", 8)
        self.assertAlmostEqual(g[16, 16], 0.0)


if __name__ == "__main__":
    unittest.main()
