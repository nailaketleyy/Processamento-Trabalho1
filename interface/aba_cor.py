"""Aba para operações de cor RGB/HSI, chroma-key e tom sépia."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser

import numpy as np
from PIL import Image

from algoritmos import cor

from .aba_base import Aba
from .widgets import slider


class AbaCor(Aba):
    titulo = "Cor RGB / HSI"

    def construir(self, f):
        ttk.Label(
            f,
            text=("Converte RGB↔HSI, equaliza a intensidade HSI e ajusta matiz, "
                  "saturação e brilho. As demais ferramentas incluem cinza, "
                  "negativo, chroma-key, canais C/R-M/G-Y/B e sépia."),
            wraplength=400, justify="left",
        ).pack(fill="x", padx=8, pady=6)

        lf = ttk.LabelFrame(f, text="Operação")
        lf.pack(fill="x", padx=6, pady=4)
        opcoes = (
            ("Converter RGB → HSI (matiz H)", "hsi_h"),
            ("Converter RGB → HSI (saturação S)", "hsi_s"),
            ("Converter RGB → HSI (intensidade I)", "hsi_i"),
            ("Converter HSI → RGB", "hsi_rgb"),
            ("Equalizar histograma da intensidade HSI", "equalizar"),
            ("Ajustar matiz, saturação e brilho", "ajustes"),
            ("Cinza: média aritmética simples", "cinza_media"),
            ("Cinza: luminância ponderada", "cinza_ponderada"),
            ("Negativo", "negativo"),
            ("Ajustar canais C/R, M/G, Y/B", "cmy"),
            ("Sépia", "sepia"),
            ("Chroma-key (cor escolhida)", "chroma"),
        )
        self.operacao = tk.StringVar(value=opcoes[0][0])
        ttk.Combobox(
            lf, textvariable=self.operacao,
            values=[label for label, _ in opcoes], state="readonly",
        ).pack(fill="x", padx=5, pady=4)
        self._op_map = {label: value for label, value in opcoes}

        lf = ttk.LabelFrame(f, text="Ajustes HSI")
        lf.pack(fill="x", padx=6, pady=4)
        self.matiz = slider(lf, "Matiz (graus)", -180, 180, 0, 1)
        self.saturacao = slider(lf, "Fator de saturação", 0, 2, 1, 0.05)
        self.brilho = slider(lf, "Fator de intensidade/brilho", 0, 2, 1, 0.05)

        lf = ttk.LabelFrame(f, text="Ajuste de canais")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            lf, text="Valores positivos reforçam R/G/B; negativos reforçam C/M/Y.",
            wraplength=400,
        ).pack(anchor="w", padx=5, pady=2)
        self.cr = slider(lf, "C (-) / R (+)", -255, 255, 0, 1)
        self.mg = slider(lf, "M (-) / G (+)", -255, 255, 0, 1)
        self.yb = slider(lf, "Y (-) / B (+)", -255, 255, 0, 1)

        lf = ttk.LabelFrame(f, text="Chroma-key")
        lf.pack(fill="x", padx=6, pady=4)
        self.cor_chave = (0, 255, 0)
        linha = ttk.Frame(lf)
        linha.pack(fill="x", padx=5, pady=3)
        ttk.Button(linha, text="Escolher cor-chave...", command=self.escolher_cor).pack(side="left")
        self.amostra = tk.Label(linha, width=6, bg="#00ff00", relief="sunken")
        self.amostra.pack(side="left", padx=6)
        self.var_espaco = tk.StringVar(value="RGB")
        ttk.Combobox(linha, textvariable=self.var_espaco, values=("RGB", "HSI"),
                     state="readonly", width=6).pack(side="left")
        self.s_distancia = slider(lf, "Tolerância (distância máxima à cor-chave)", 0, 442, 100, 1)
        self.fundo_path = tk.StringVar(value="Imagem substituta: não selecionada")
        ttk.Label(lf, textvariable=self.fundo_path, wraplength=400).pack(anchor="w", padx=5)
        ttk.Button(lf, text="Escolher imagem substituta...", command=self.escolher_fundo).pack(
            fill="x", padx=5, pady=3)

        ttk.Button(
            f, text="Pré-visualizar",
            command=lambda: self._executar(True),
        ).pack(fill="x", padx=8, pady=(8, 3))
        ttk.Button(
            f, text="Aplicar",
            command=lambda: self._executar(False),
        ).pack(fill="x", padx=8, pady=3)
        self._fundo = None

    def escolher_cor(self):
        res = colorchooser.askcolor(color="#%02x%02x%02x" % self.cor_chave,
                                    parent=self.app.root, title="Cor-chave")
        if res and res[0]:
            self.cor_chave = tuple(int(round(v)) for v in res[0])
            self.amostra.configure(bg=res[1])

    def escolher_fundo(self):
        caminho = filedialog.askopenfilename(
            title="Imagem para substituir o verde",
            filetypes=[("Imagens", "*.bmp *.jpg *.jpeg *.png"), ("Todos", "*.*")],
        )
        if not caminho:
            return
        try:
            self._fundo = np.array(Image.open(caminho).convert("RGB"), dtype=np.uint8)
        except (OSError, ValueError) as erro:
            messagebox.showerror("Chroma-key", f"Não foi possível abrir a imagem:\n{erro}",
                                 parent=self.app.root)
            return
        self.fundo_path.set(caminho)

    def fn_aplicar(self):
        img = self.app.atual
        operacao = self._op_map[self.operacao.get()]
        if operacao == "hsi_h":
            return cor.componente_hsi(img, "H")
        if operacao == "hsi_s":
            return cor.componente_hsi(img, "S")
        if operacao == "hsi_i":
            return cor.componente_hsi(img, "I")
        if operacao == "hsi_rgb":
            return cor.hsi_para_rgb(*cor.rgb_para_hsi(img))
        if operacao == "equalizar":
            return cor.equalizar_hsi(img)
        if operacao == "ajustes":
            return cor.ajustar_hsi(
                img, float(self.matiz.get()), float(self.saturacao.get()),
                float(self.brilho.get()))
        if operacao == "cinza_media":
            return cor.escala_cinza(img, "media")
        if operacao == "cinza_ponderada":
            return cor.escala_cinza(img, "ponderada")
        if operacao == "negativo":
            return cor.negativo(img)
        if operacao == "cmy":
            return cor.ajustar_cmy(
                img, float(self.cr.get()), float(self.mg.get()), float(self.yb.get()))
        if operacao == "sepia":
            return cor.sepia(img)
        if operacao == "chroma":
            if self._fundo is None:
                raise ValueError("Escolha a imagem que substituirá a cor-chave.")
            return cor.chroma_key(img, self._fundo, float(self.s_distancia.get()),
                                  self.cor_chave, self.var_espaco.get().lower())
        raise ValueError("Operação de cor desconhecida.")

    def _executar(self, previsualizar):
        if not self.app.precisa_imagem():
            return
        try:
            if previsualizar:
                self.app.prever(self.fn_aplicar)
            else:
                self.app.aplicar(self.fn_aplicar)
        except ValueError as erro:
            messagebox.showerror("Operação de cor", str(erro), parent=self.app.root)
