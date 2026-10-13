"""ETAPA 2 - Aba: Histograma e equalização de histograma."""

import tkinter as tk
from tkinter import ttk

import numpy as np

from algoritmos import histograma as hg
from algoritmos.base import cinza

from .aba_base import Aba

CORES = {"Cinza": "#404040", "R": "#d62728", "G": "#2ca02c", "B": "#1f77b4"}


class AbaHistograma(Aba):
    titulo = "Histograma"
    LARG, ALT = 290, 125

    def construir(self, f):
        app = self.app
        self._cache = {}

        lf = ttk.LabelFrame(f, text="Histograma")
        lf.pack(fill="x", padx=6, pady=4)
        linha = ttk.Frame(lf)
        linha.pack(fill="x", padx=4)
        self.var_canal = tk.StringVar(value="Cinza")
        for c in ("Cinza", "R", "G", "B", "RGB"):
            ttk.Radiobutton(linha, text=c, value=c, variable=self.var_canal,
                            command=self.redesenhar).pack(side="left", padx=2)
        self.var_log = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Escala logarítmica", variable=self.var_log,
                        command=self.redesenhar).pack(anchor="w", padx=4)

        ttk.Label(lf, text="Original").pack()
        self.cv_h1 = tk.Canvas(lf, width=self.LARG, height=self.ALT, bg="white", highlightthickness=1)
        self.cv_h1.pack(padx=4, pady=(0, 4))
        ttk.Label(lf, text="Resultado").pack()
        self.cv_h2 = tk.Canvas(lf, width=self.LARG, height=self.ALT, bg="white", highlightthickness=1)
        self.cv_h2.pack(padx=4, pady=(0, 4))

        lf = ttk.LabelFrame(f, text="Equalização de histograma")
        lf.pack(fill="x", padx=6, pady=4)
        self.var_eq = tk.StringVar(value="lum")
        for texto, valor in (("Luminância (preserva as cores)", "lum"),
                             ("Cada canal RGB separadamente", "rgb"),
                             ("Converter para tons de cinza", "cinza")):
            ttk.Radiobutton(lf, text=texto, value=valor, variable=self.var_eq,
                            command=lambda: app.prever(self.fn_equalizar)).pack(anchor="w", padx=6)
        ttk.Button(lf, text="Equalizar histograma",
                   command=lambda: app.aplicar(self.fn_equalizar)).pack(pady=6)

        app.ao_atualizar(self.redesenhar)

    def fn_equalizar(self):
        return hg.equalizar(self.app.atual, self.var_eq.get())

    # ---------------- desenho ----------------
    def _hists(self, slot, img):
        """Calcula os histogramas uma única vez por matriz de imagem."""
        ent = self._cache.get(slot)
        if ent is not None and ent[0] is img:
            return ent[1]
        h = {"Cinza": hg.histograma(cinza(img)),
             "R": hg.histograma(img[..., 0]),
             "G": hg.histograma(img[..., 1]),
             "B": hg.histograma(img[..., 2])}
        self._cache[slot] = (img, h)
        return h

    def redesenhar(self):
        if not self.visivel():
            return  # só desenha quando a aba está visível
        self._desenhar(self.cv_h1, "orig", self.app.original)
        self._desenhar(self.cv_h2, "res", self.app.resultado_atual())

    def _desenhar(self, cv, slot, img):
        cv.delete("all")
        W, H = self.LARG, self.ALT
        if img is None:
            cv.create_text(W // 2, H // 2, text="(sem imagem)", fill="gray")
            return
        hs = self._hists(slot, img)
        canal = self.var_canal.get()
        series = [("R", CORES["R"]), ("G", CORES["G"]), ("B", CORES["B"])] if canal == "RGB" \
            else [(canal, CORES[canal])]
        dados = []
        for c, cor in series:
            v = hs[c].astype(np.float64)
            dados.append((np.log1p(v) if self.var_log.get() else v, cor))
        pico = max(d.max() for d, _ in dados) or 1.0

        x0, base, topo = 6, H - 16, 6
        pw, ph = W - 12, base - topo
        for d, cor in dados:
            if len(dados) == 1:   # barras
                for k in range(256):
                    if d[k] > 0:
                        x = x0 + k * pw / 255
                        cv.create_line(x, base, x, base - d[k] / pico * ph, fill=cor)
            else:                 # linhas sobrepostas
                pts = []
                for k in range(256):
                    pts += [x0 + k * pw / 255, base - d[k] / pico * ph]
                cv.create_line(*pts, fill=cor)
        cv.create_line(x0, base, x0 + pw, base)
        cv.create_text(x0, base + 8, text="0", anchor="w", font=("", 8))
        cv.create_text(x0 + pw, base + 8, text="255", anchor="e", font=("", 8))
        media, desvio = hg.estatisticas(hs["Cinza"])
        cv.create_text(W - 6, 8, anchor="ne", font=("", 8),
                       text=f"média {media:.1f}   desvio {desvio:.1f}")
