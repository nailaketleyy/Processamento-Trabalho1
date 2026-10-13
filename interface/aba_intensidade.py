"""ETAPA 1 - Abas: Intensidade e Linear por partes."""

import tkinter as tk
from tkinter import ttk

from algoritmos import intensidade as ti
from algoritmos.base import aplicar_lut, cinza

from .aba_base import Aba
from .widgets import slider


class AbaIntensidade(Aba):
    titulo = "Intensidade"

    def construir(self, f):
        app = self.app

        lf = ttk.LabelFrame(f, text="Negativo   s = L - 1 - r")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Button(lf, text="Aplicar negativo",
                   command=lambda: app.aplicar(self.fn_negativo)).pack(pady=4)

        lf = ttk.LabelFrame(f, text="Brilho   s = r + b")
        lf.pack(fill="x", padx=6, pady=4)
        self.s_brilho = slider(lf, "b", -255, 255, 0, 1, lambda: app.prever(self.fn_brilho))
        ttk.Button(lf, text="Aplicar brilho",
                   command=lambda: app.aplicar(self.fn_brilho)).pack(pady=4)

        lf = ttk.LabelFrame(f, text="Potência (gama)   s = c · r^γ")
        lf.pack(fill="x", padx=6, pady=4)
        self.s_gc = slider(lf, "c", 0.1, 3.0, 1.0, 0.05, lambda: app.prever(self.fn_gama))
        self.s_gg = slider(lf, "γ (gama)", 0.04, 5.0, 1.0, 0.01, lambda: app.prever(self.fn_gama))
        ttk.Button(lf, text="Aplicar gama",
                   command=lambda: app.aplicar(self.fn_gama)).pack(pady=4)

        lf = ttk.LabelFrame(f, text="Limiarização")
        lf.pack(fill="x", padx=6, pady=4)
        self.s_lim = slider(lf, "Limiar T", 0, 255, 128, 1, lambda: app.prever(self.fn_limiar))
        ttk.Button(lf, text="Aplicar limiarização",
                   command=lambda: app.aplicar(self.fn_limiar)).pack(pady=4)

    def fn_negativo(self):
        return aplicar_lut(self.app.atual, ti.lut_negativo())

    def fn_brilho(self):
        return aplicar_lut(self.app.atual, ti.lut_brilho(int(self.s_brilho.get())))

    def fn_gama(self):
        return aplicar_lut(self.app.atual,
                           ti.lut_gama(float(self.s_gc.get()), float(self.s_gg.get())))

    def fn_limiar(self):
        return ti.limiarizar(self.app.atual, int(self.s_lim.get()))


class AbaLinearPartes(Aba):
    titulo = "Linear por partes"

    def construir(self, f):
        app = self.app

        lf = ttk.LabelFrame(f, text="Alargamento de contraste")
        lf.pack(fill="x", padx=6, pady=4)
        self.cv_curva = tk.Canvas(lf, width=150, height=150, bg="white", highlightthickness=1)
        self.cv_curva.pack(pady=2)
        self.s_r1 = slider(lf, "r1", 0, 255, 70, 1, self._mudou_contraste)
        self.s_s1 = slider(lf, "s1", 0, 255, 0, 1, self._mudou_contraste)
        self.s_r2 = slider(lf, "r2", 0, 255, 180, 1, self._mudou_contraste)
        self.s_s2 = slider(lf, "s2", 0, 255, 255, 1, self._mudou_contraste)
        bt = ttk.Frame(lf)
        bt.pack(pady=4)
        ttk.Button(bt, text="Auto (r_min, r_max)", command=self.auto_contraste).pack(side="left", padx=2)
        ttk.Button(bt, text="Aplicar",
                   command=lambda: app.aplicar(self.fn_contraste)).pack(side="left", padx=2)
        self._desenhar_curva()

        lf = ttk.LabelFrame(f, text="Fatiamento de níveis de intensidade")
        lf.pack(fill="x", padx=6, pady=4)
        self.s_fa = slider(lf, "A", 0, 255, 100, 1, lambda: app.prever(self.fn_fatiamento))
        self.s_fb = slider(lf, "B", 0, 255, 180, 1, lambda: app.prever(self.fn_fatiamento))
        self.var_preserva = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Preservar demais níveis (senão, fundo escuro)",
                        variable=self.var_preserva,
                        command=lambda: app.prever(self.fn_fatiamento)).pack(anchor="w", padx=6)
        ttk.Button(lf, text="Aplicar fatiamento",
                   command=lambda: app.aplicar(self.fn_fatiamento)).pack(pady=4)

        lf = ttk.LabelFrame(f, text="Planos de bits (1 = menos significativo)")
        lf.pack(fill="x", padx=6, pady=4)
        self.s_plano = slider(lf, "Plano", 1, 8, 8, 1, lambda: app.prever(self.fn_plano))
        ttk.Button(lf, text="Aplicar plano de bits",
                   command=lambda: app.aplicar(self.fn_plano)).pack(pady=4)

    def _mudou_contraste(self):
        self._desenhar_curva()
        self.app.prever(self.fn_contraste)

    def _pontos_contraste(self):
        r1, s1 = int(self.s_r1.get()), int(self.s_s1.get())
        r2, s2 = int(self.s_r2.get()), int(self.s_s2.get())
        r2, s2 = max(r1, r2), max(s1, s2)  # mantém T(r) monotonicamente crescente
        return [(0, 0), (r1, s1), (r2, s2), (255, 255)]

    def _desenhar_curva(self):
        c = self.cv_curva
        c.delete("all")
        pts = self._pontos_contraste()
        m, t = 8, 134
        xy = []
        for r, s in pts:
            xy += [m + r * t / 255, m + t - s * t / 255]
        c.create_line(m, m + t, m + t, m, fill="#cccccc", dash=(2, 2))  # identidade
        c.create_line(*xy, fill="#1a5fb4", width=2)
        for i in (1, 2):
            c.create_oval(xy[2 * i] - 3, xy[2 * i + 1] - 3,
                          xy[2 * i] + 3, xy[2 * i + 1] + 3, fill="#c01c28", outline="")

    def auto_contraste(self):
        """Estica o intervalo [r_min, r_max] da imagem para [0, 255]."""
        if not self.app.precisa_imagem():
            return
        g = cinza(self.app.atual)
        self.s_r1.set(int(g.min()))
        self.s_s1.set(0)
        self.s_r2.set(int(g.max()))
        self.s_s2.set(255)
        self._mudou_contraste()

    def fn_contraste(self):
        return aplicar_lut(self.app.atual, ti.lut_linear_partes(self._pontos_contraste()))

    def fn_fatiamento(self):
        a, b = int(self.s_fa.get()), int(self.s_fb.get())
        return aplicar_lut(self.app.atual,
                           ti.lut_fatiamento(min(a, b), max(a, b), self.var_preserva.get()))

    def fn_plano(self):
        return ti.plano_de_bits(self.app.atual, int(self.s_plano.get()))
