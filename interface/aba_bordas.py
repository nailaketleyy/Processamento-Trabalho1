"""Aba de detecção de bordas por Sobel e magnitude do gradiente."""

import tkinter as tk
from tkinter import ttk

from algoritmos import bordas

from .aba_base import Aba


class AbaBordas(Aba):
    titulo = "Sobel / Gradiente"

    def construir(self, f):
        ttk.Label(
            f,
            text=("Sobel X e Y mostram as respostas direcionais normalizadas. "
                  "Gradiente combina as duas por magnitude."),
            wraplength=330,
            justify="left",
        ).pack(fill="x", padx=8, pady=8)

        self.var_operacao = tk.StringVar(value="gradiente")
        lf = ttk.LabelFrame(f, text="Operação")
        lf.pack(fill="x", padx=6, pady=4)
        for label, value in (
            ("Sobel X (bordas verticais)", "x"),
            ("Sobel Y (bordas horizontais)", "y"),
            ("Magnitude do gradiente", "gradiente"),
        ):
            ttk.Radiobutton(
                lf, text=label, value=value, variable=self.var_operacao
            ).pack(anchor="w", padx=6, pady=2)

        self.var_borda = tk.StringVar(value="replicar")
        ttk.Label(lf, text="Tratamento da borda:").pack(anchor="w", padx=6, pady=(6, 0))
        ttk.Combobox(
            lf, textvariable=self.var_borda,
            values=("zeros", "replicar", "espelhar"), state="readonly",
        ).pack(fill="x", padx=6, pady=(0, 6))

        ttk.Button(
            f, text="Pré-visualizar",
            command=lambda: self.app.prever(self.fn_aplicar),
        ).pack(fill="x", padx=8, pady=(8, 3))
        ttk.Button(
            f, text="Aplicar",
            command=lambda: self.app.aplicar(self.fn_aplicar),
        ).pack(fill="x", padx=8, pady=3)

    def fn_aplicar(self):
        operacao = self.var_operacao.get()
        if operacao == "gradiente":
            return bordas.gradiente(self.app.atual, self.var_borda.get())
        return bordas.sobel(self.app.atual, operacao, self.var_borda.get())
