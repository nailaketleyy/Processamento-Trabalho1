"""Aba de aguçamento: Laplaciano (4/8 vizinhos) e máscara de nitidez / high-boost."""

import tkinter as tk
from tkinter import ttk, messagebox

from algoritmos import convolucao as cv

from .aba_base import Aba
from .widgets import slider

BORDAS = {"Zeros": "zeros", "Replicar borda": "replicar", "Espelhar": "espelhar"}


class AbaNitidez(Aba):
    titulo = "Nitidez"

    def construir(self, f):
        self.var_borda = tk.StringVar(value="Replicar borda")
        ttk.Label(f, text="Borda:").pack(anchor="w", padx=8, pady=(6, 0))
        ttk.Combobox(f, textvariable=self.var_borda, values=list(BORDAS),
                     state="readonly").pack(fill="x", padx=8)

        lf = ttk.LabelFrame(f, text="Laplaciano")
        lf.pack(fill="x", padx=6, pady=4)
        self.var_viz = tk.IntVar(value=4)
        ttk.Radiobutton(lf, text="4 vizinhos", value=4, variable=self.var_viz).pack(anchor="w", padx=6)
        ttk.Radiobutton(lf, text="8 vizinhos", value=8, variable=self.var_viz).pack(anchor="w", padx=6)
        self.var_sinal = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Centro negativo (g = f − ∇²f); desmarcado: centro positivo (g = f + ∇²f)",
                        variable=self.var_sinal).pack(anchor="w", padx=6)
        ttk.Button(lf, text="Mostrar Laplaciano (normalizado)",
                   command=lambda: self._executar(self.fn_lap, True)).pack(fill="x", padx=6, pady=2)
        ttk.Button(lf, text="Aplicar aguçamento Laplaciano",
                   command=lambda: self._executar(self.fn_agucar, False)).pack(fill="x", padx=6, pady=2)

        lf = ttk.LabelFrame(f, text="Máscara de nitidez / high-boost   g = f + k(f − f_suave)")
        lf.pack(fill="x", padx=6, pady=4)
        self.var_tipo = tk.StringVar(value="gaussiano")
        ttk.Radiobutton(lf, text="Suavização gaussiana", value="gaussiano", variable=self.var_tipo).pack(anchor="w", padx=6)
        ttk.Radiobutton(lf, text="Suavização por média", value="media", variable=self.var_tipo).pack(anchor="w", padx=6)
        self.s_n = slider(lf, "Tamanho do filtro (ímpar)", 3, 21, 5, 2)
        self.s_sigma = slider(lf, "σ (gaussiano)", 0.3, 8.0, 1.0, 0.1)
        self.s_k = slider(lf, "k (1 = unsharp; >1 = high-boost)", 0.0, 6.0, 1.0, 0.1)
        ttk.Button(lf, text="Pré-visualizar", command=lambda: self._executar(self.fn_unsharp, True)
                   ).pack(fill="x", padx=6, pady=2)
        ttk.Button(lf, text="Aplicar", command=lambda: self._executar(self.fn_unsharp, False)
                   ).pack(fill="x", padx=6, pady=2)

    def _borda(self):
        return BORDAS[self.var_borda.get()]

    def fn_lap(self):
        return cv.laplaciano_normalizado(self.app.atual, self.var_viz.get(),
                                         self.var_sinal.get(), self._borda())

    def fn_agucar(self):
        return cv.agucar_laplaciano(self.app.atual, self.var_viz.get(),
                                    self.var_sinal.get(), self._borda())

    def fn_unsharp(self):
        return cv.unsharp_highboost(self.app.atual, float(self.s_k.get()), self.var_tipo.get(),
                                    int(self.s_n.get()), float(self.s_sigma.get()), self._borda())

    def _executar(self, fn, previa):
        if not self.app.precisa_imagem():
            return
        try:
            (self.app.prever if previa else self.app.aplicar)(fn)
        except ValueError as erro:
            messagebox.showerror("Nitidez", str(erro), parent=self.app.root)
