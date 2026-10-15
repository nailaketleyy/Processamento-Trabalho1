"""ETAPA 2 - Aba: Filtragem espacial por convolução (filtro genérico n x n)."""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np

from algoritmos import convolucao as cv

from .aba_base import Aba
from .widgets import slider

PREDEFINIDOS = [
    "Média (caixa)", "Gaussiano", "Mediana", "Aguçamento Laplaciano",
    "High-boost", "Identidade",
] + list(cv.KERNELS_3X3) + ["Personalizado"]
TAMANHOS = [str(n) for n in range(3, 22, 2)]            # 3, 5, ..., 21
BORDAS = {"Zeros": "zeros", "Replicar borda": "replicar", "Espelhar": "espelhar"}
SAIDAS = {"Saturar em [0, 255]": "saturar", "Valor absoluto": "absoluto",
          "Normalizar [mín, máx] → [0, 255]": "normalizar"}


class AbaConvolucao(Aba):
    titulo = "Convolução"

    def construir(self, f):
        app = self.app
        self.kernel = cv.kernel_media(3)

        lf = ttk.LabelFrame(f, text="Filtro")
        lf.pack(fill="x", padx=6, pady=4)
        grade = ttk.Frame(lf)
        grade.pack(fill="x", padx=6, pady=2)
        ttk.Label(grade, text="Predefinido:").grid(row=0, column=0, sticky="w")
        self.var_preset = tk.StringVar(value=PREDEFINIDOS[0])
        c = ttk.Combobox(grade, textvariable=self.var_preset, values=PREDEFINIDOS,
                         state="readonly", width=22)
        c.grid(row=0, column=1, sticky="w", padx=4, pady=2)
        c.bind("<<ComboboxSelected>>", lambda e: self._mudou_preset())

        ttk.Label(grade, text="Tamanho n × n:").grid(row=1, column=0, sticky="w")
        self.var_tam = tk.StringVar(value="3")
        c = ttk.Combobox(grade, textvariable=self.var_tam, values=TAMANHOS,
                         state="readonly", width=6)
        c.grid(row=1, column=1, sticky="w", padx=4, pady=2)
        c.bind("<<ComboboxSelected>>", lambda e: self._mudou_tamanho())

        self.s_sigma = slider(lf, "σ (gaussiano / high-boost)", 0.3, 8.0, 1.0, 0.1,
                              self._mudou_sigma)
        self.s_boost = slider(lf, "Fator A do high-boost", 1.0, 5.0, 1.5, 0.1,
                              self._mudou_parametro)
        self.bt_editar = ttk.Button(lf, text="Ver / editar coeficientes...", command=self.editar)
        self.bt_editar.pack(pady=2)
        self.lbl_resumo = ttk.Label(lf, text="", wraplength=290, justify="left")
        self.lbl_resumo.pack(padx=6, pady=(0, 4))

        lf = ttk.LabelFrame(f, text="Opções")
        lf.pack(fill="x", padx=6, pady=4)
        self.var_op = tk.StringVar(value="conv")
        ttk.Radiobutton(lf, text="Convolução", value="conv", variable=self.var_op).pack(anchor="w", padx=6)
        ttk.Radiobutton(lf, text="Correlação", value="corr", variable=self.var_op).pack(anchor="w", padx=6)
        self.var_norm = tk.BooleanVar(value=True)
        self.ck_norm = ttk.Checkbutton(
            lf, text="Dividir pela soma dos coeficientes (se ≠ 0)",
            variable=self.var_norm)
        self.ck_norm.pack(anchor="w", padx=6, pady=(4, 2))

        grade = ttk.Frame(lf)
        grade.pack(fill="x", padx=6, pady=2)
        ttk.Label(grade, text="Borda:").grid(row=0, column=0, sticky="w")
        self.var_borda = tk.StringVar(value="Zeros")
        ttk.Combobox(grade, textvariable=self.var_borda, values=list(BORDAS),
                     state="readonly", width=22).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(grade, text="Saída:").grid(row=1, column=0, sticky="w")
        self.var_saida = tk.StringVar(value="Saturar em [0, 255]")
        ttk.Combobox(grade, textvariable=self.var_saida, values=list(SAIDAS),
                     state="readonly", width=22).grid(row=1, column=1, padx=4, pady=2)

        bt = ttk.Frame(f)
        bt.pack(pady=6)
        ttk.Button(bt, text="Pré-visualizar", command=lambda: app.prever(self.fn_filtro)).pack(side="left", padx=4)
        ttk.Button(bt, text="Aplicar filtro", command=lambda: app.aplicar(self.fn_filtro)).pack(side="left", padx=4)

        self._gerar()

    # ---------------- construção do kernel ----------------
    def _n(self):
        return int(self.var_tam.get())

    def _gerar(self):
        """Gera o kernel do predefinido atual no tamanho atual."""
        nome, n = self.var_preset.get(), self._n()
        if nome == "Média (caixa)":
            self.kernel = cv.kernel_media(n)
        elif nome == "Gaussiano":
            self.kernel = cv.kernel_gaussiano(n, float(self.s_sigma.get()))
        elif nome == "Aguçamento Laplaciano":
            self.kernel = cv.kernel_aguçamento_laplaciano(n)
        elif nome == "High-boost":
            self.kernel = cv.kernel_high_boost(n, float(self.s_boost.get()),
                                               float(self.s_sigma.get()))
        elif nome == "Identidade":
            self.kernel = cv.kernel_identidade(n)
        elif nome in cv.KERNELS_3X3:
            self.kernel = np.array(cv.KERNELS_3X3[nome][0], dtype=float)
        self._atualizar_controles()
        self._resumo()

    def _atualizar_controles(self):
        nome = self.var_preset.get()
        if nome in ("Mediana", "High-boost"):
            self.bt_editar.state(["disabled"])
            self.ck_norm.state(["disabled"])
        else:
            self.bt_editar.state(["!disabled"])
            self.ck_norm.state(["!disabled"])

    def _resumo(self):
        if self.var_preset.get() == "Mediana":
            self.lbl_resumo.config(
                text=f"Filtro mediana {self._n()}×{self._n()} · não linear; sem kernel de convolução")
            return
        n = self.kernel.shape[0]
        extra = f" · fator A = {self.s_boost.get():g}" if self.var_preset.get() == "High-boost" else ""
        self.lbl_resumo.config(
            text=f"Filtro {n}×{n} · soma dos coeficientes = {self.kernel.sum():g}{extra}")

    def _mudou_preset(self):
        nome = self.var_preset.get()
        if nome in cv.KERNELS_3X3:
            self.var_tam.set("3")
            self.var_saida.set("Valor absoluto" if cv.KERNELS_3X3[nome][1] == "absoluto"
                               else "Saturar em [0, 255]")
        elif nome != "Personalizado":
            self.var_saida.set("Saturar em [0, 255]")
        if nome == "High-boost":
            self.var_norm.set(False)
        self._atualizar_controles()
        if nome == "Mediana":
            self._resumo()
        elif nome == "Personalizado":
            self._resumo()
        else:
            self._gerar()

    def _mudou_tamanho(self):
        nome, n = self.var_preset.get(), self._n()
        if nome in ("Média (caixa)", "Gaussiano", "Mediana",
                    "Aguçamento Laplaciano", "High-boost", "Identidade"):
            self._gerar()
        else:  # filtros fixos/personalizado: corta ou completa com zeros
            self.kernel = cv.redimensionar_kernel(self.kernel, n)
            if nome in cv.KERNELS_3X3 and n != 3:
                self.var_preset.set("Personalizado")
            self._resumo()

    def _mudou_sigma(self):
        if self.var_preset.get() in ("Gaussiano", "High-boost"):
            self._gerar()

    def _mudou_parametro(self):
        if self.var_preset.get() == "High-boost":
            self._gerar()

    # ---------------- editor de coeficientes ----------------
    def editar(self):
        n = self.kernel.shape[0]
        win = tk.Toplevel(self.app.root)
        win.title(f"Coeficientes do filtro {n}×{n}")
        win.rowconfigure(0, weight=1)
        win.columnconfigure(0, weight=1)

        tela = tk.Canvas(win, highlightthickness=0)
        sy = ttk.Scrollbar(win, orient="vertical", command=tela.yview)
        sx = ttk.Scrollbar(win, orient="horizontal", command=tela.xview)
        tela.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        tela.grid(row=0, column=0, sticky="nsew")
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")

        interno = ttk.Frame(tela)
        tela.create_window((0, 0), window=interno, anchor="nw")
        interno.bind("<Configure>", lambda e: tela.configure(scrollregion=tela.bbox("all")))

        entradas = []
        for i in range(n):
            linha = []
            for j in range(n):
                e = tk.Entry(interno, width=8, justify="center")
                e.insert(0, f"{self.kernel[i, j]:g}")
                e.grid(row=i, column=j, padx=1, pady=1)
                linha.append(e)
            entradas.append(linha)

        def ok():
            try:
                k = np.array([[float(entradas[i][j].get().replace(",", "."))
                               for j in range(n)] for i in range(n)])
            except ValueError:
                messagebox.showerror("Filtro", "Há coeficientes que não são números.", parent=win)
                return
            self.kernel = k
            self.var_preset.set("Personalizado")
            self._resumo()
            win.destroy()

        def zerar():
            for linha in entradas:
                for e in linha:
                    e.delete(0, "end")
                    e.insert(0, "0")

        bt = ttk.Frame(win)
        bt.grid(row=2, column=0, columnspan=2, pady=6)
        ttk.Button(bt, text="Zerar tudo", command=zerar).pack(side="left", padx=4)
        ttk.Button(bt, text="Usar estes coeficientes", command=ok).pack(side="left", padx=4)
        ttk.Button(bt, text="Cancelar", command=win.destroy).pack(side="left", padx=4)

        win.update_idletasks()
        win.geometry(f"{min(interno.winfo_reqwidth() + 30, 950)}x"
                     f"{min(interno.winfo_reqheight() + 80, 650)}")

    # ---------------- aplicação ----------------
    def fn_filtro(self):
        if self.var_preset.get() == "Mediana":
            return cv.mediana(self.app.atual, self._n(),
                              borda=BORDAS[self.var_borda.get()])
        k = self.kernel.copy()
        if self.var_norm.get() and self.var_preset.get() != "High-boost":
            soma = k.sum()
            if soma != 0:
                k = k / soma
        return cv.filtrar(self.app.atual, k,
                          usar_convolucao=(self.var_op.get() == "conv"),
                          borda=BORDAS[self.var_borda.get()],
                          saida=SAIDAS[self.var_saida.get()])
