"""Ferramentas de Fourier, filtros ideais, edição do espectro e aliasing."""

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
from PIL import Image, ImageTk

from algoritmos import frequencia as fq
from algoritmos.base import cinza, cinza_para_rgb
from algoritmos.convolucao import kernel_media

from .aba_base import Aba
from .widgets import slider


FILTROS = ("passa-baixa", "rejeita-baixa", "passa-faixa", "rejeita-faixa",
           "passa-notch", "rejeita-notch")


class AbaFrequencia(Aba):
    titulo = "Frequência / Aliasing"

    def construir(self, f):
        self._imagem_fonte = None
        self._espectro = None
        self._mascara = None
        self._metodo_espectro = None
        self._foto_espectro = None
        self._display_shape = (0, 0)
        self._display_offset = (0, 0)
        ttk.Label(
            f,
            text=("O processamento de frequência usa a luminância. DFT direta "
                  "é educativa e limitada a 4096 pixels; FFT NumPy é recomendada "
                  "para imagens maiores."),
            wraplength=330, justify="left",
        ).pack(fill="x", padx=8, pady=6)

        lf = ttk.LabelFrame(f, text="Espectro e filtro ideal")
        lf.pack(fill="x", padx=6, pady=4)
        linha = ttk.Frame(lf)
        linha.pack(fill="x", padx=5, pady=3)
        ttk.Label(linha, text="Transformada:").pack(side="left")
        self.var_metodo = tk.StringVar(value="fft")
        ttk.Combobox(
            linha, textvariable=self.var_metodo, values=("fft", "dft"),
            state="readonly", width=8,
        ).pack(side="left", padx=4)
        ttk.Button(linha, text="Calcular espectro", command=self.calcular).pack(side="left")
        self.var_saida = tk.StringVar(value="normalizar")
        ttk.Label(lf, text="Exibição da imagem reconstruída:").pack(anchor="w", padx=5)
        for texto, valor in (("Saturar [0,255]", "saturar"),
                             ("Normalizar [mín,máx]", "normalizar")):
            ttk.Radiobutton(
                lf, text=texto, value=valor, variable=self.var_saida,
            ).pack(anchor="w", padx=5)

        self.var_filtro = tk.StringVar(value="passa-baixa")
        ttk.Label(lf, text="Filtro ideal:").pack(anchor="w", padx=5)
        ttk.Combobox(
            lf, textvariable=self.var_filtro, values=FILTROS,
            state="readonly",
        ).pack(fill="x", padx=5)
        self.s_corte = slider(lf, "Corte (baixa/alta)", 0, 256, 24, 1)
        self.s_interno = slider(lf, "Raio interno (faixa)", 0, 256, 10, 1)
        self.s_externo = slider(lf, "Raio externo (faixa)", 0, 256, 40, 1)
        self.s_notch_x = slider(lf, "Centro notch u (deslocamento vertical)", -128, 128, 30, 1)
        self.s_notch_y = slider(lf, "Centro notch v (deslocamento horizontal)", -128, 128, 30, 1)
        self.s_notch_r = slider(lf, "Raio do notch", 0, 64, 5, 1)
        ttk.Button(lf, text="Criar máscara ideal", command=self.criar_mascara).pack(pady=3)

        lf = ttk.LabelFrame(f, text="Editar o espectro deslocado")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            lf, text="Arraste sobre o espectro. Preto rejeita; branco mantém; "
                     "cinza atenua parcialmente.",
            wraplength=330, justify="left",
        ).pack(anchor="w", padx=5, pady=3)
        self.cv_espectro = tk.Canvas(
            lf, width=330, height=230, background="black", highlightthickness=1)
        self.cv_espectro.pack(padx=4, pady=3)
        self.cv_espectro.bind("<Button-1>", self._pintar_evento)
        self.cv_espectro.bind("<B1-Motion>", self._pintar_evento)
        linha = ttk.Frame(lf)
        linha.pack(fill="x")
        self.var_pincel = tk.DoubleVar(value=0.0)
        ttk.Label(linha, text="Nível do pincel").pack(side="left", padx=4)
        ttk.Combobox(
            linha, textvariable=self.var_pincel,
            values=(0.0, 0.25, 0.5, 0.75, 1.0), state="readonly", width=6,
        ).pack(side="left", padx=4)
        self.var_suave = tk.BooleanVar(value=True)
        ttk.Checkbutton(linha, text="Pincel suave", variable=self.var_suave).pack(side="left")
        self.s_raio = slider(lf, "Raio do pincel (pixels no espectro)", 1, 50, 12, 1)
        botoes = ttk.Frame(lf)
        botoes.pack(fill="x", pady=3)
        ttk.Button(botoes, text="Redefinir máscara", command=self.redefinir_mascara).pack(
            side="left", padx=3)
        ttk.Button(botoes, text="Aplicar filtro / IDFT", command=self.aplicar_filtro).pack(
            side="left", padx=3)

        lf = ttk.LabelFrame(f, text="Teorema da convolução")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            lf, text="Demonstra convolução circular imagem × kernel no domínio "
                     "da frequência, usando um kernel média 3×3.",
            wraplength=330, justify="left",
        ).pack(anchor="w", padx=5, pady=3)
        ttk.Button(lf, text="Aplicar convolução circular via Fourier",
                   command=self.aplicar_teorema).pack(pady=4)

        lf = ttk.LabelFrame(f, text="Aliasing e Moiré")
        lf.pack(fill="x", padx=6, pady=4)
        self.s_fator = slider(lf, "Fator de subamostragem", 2, 12, 3, 1)
        self.var_antialias = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            lf, text="Aplicar passa-baixa antes da subamostragem",
            variable=self.var_antialias,
        ).pack(anchor="w", padx=5)
        ttk.Button(lf, text="Demonstrar aliasing na imagem atual",
                   command=self.aplicar_aliasing).pack(fill="x", padx=5, pady=3)
        ttk.Button(lf, text="Gerar padrão de Moiré",
                   command=self.aplicar_moire).pack(fill="x", padx=5, pady=3)

        self.app.ao_atualizar(self._ao_atualizar_imagem)

    def _ao_atualizar_imagem(self):
        img = self.app.atual
        if img is not self._imagem_fonte:
            self._imagem_fonte = img
            self._espectro = None
            self._mascara = None
            self._metodo_espectro = None
            self.cv_espectro.delete("all")
            self.cv_espectro.create_text(
                165, 115, text="Calcule o espectro após abrir uma imagem",
                fill="white", width=300,
            )

    def calcular(self):
        if not self.app.precisa_imagem():
            return
        try:
            g = cinza(self.app.atual)
            metodo = self.var_metodo.get()
            espectro = fq.transformar(g, metodo)
            self._espectro = espectro
            self._mascara = np.ones(g.shape, dtype=np.float64)
            self._metodo_espectro = metodo
            self._imagem_fonte = self.app.atual
            self._desenhar_espectro()
        except ValueError as erro:
            self._espectro = None
            self._mascara = None
            self._metodo_espectro = None
            messagebox.showerror("Transformada de Fourier", str(erro), parent=self.app.root)

    def _desenhar_espectro(self):
        if self._espectro is None:
            return
        visual = fq.espectro_exibicao(self._espectro).astype(np.float64)
        visual = np.floor(visual * self._mascara + 0.5).astype(np.uint8)
        h, w = visual.shape
        escala = min(330 / w, 230 / h)
        display_w = max(1, int(round(w * escala)))
        display_h = max(1, int(round(h * escala)))
        self._display_shape = (display_h, display_w)
        self._display_offset = ((330 - display_w) // 2, (230 - display_h) // 2)
        imagem = Image.fromarray(visual).resize(
            (display_w, display_h), Image.Resampling.NEAREST)
        self._foto_espectro = ImageTk.PhotoImage(imagem)
        self.cv_espectro.delete("all")
        self.cv_espectro.create_image(
            self._display_offset[0], self._display_offset[1],
            image=self._foto_espectro, anchor="nw")

    def _pintar_evento(self, evento):
        if self._mascara is None:
            return
        h, w = self._mascara.shape
        display_h, display_w = self._display_shape
        offset_x, offset_y = self._display_offset
        x_display, y_display = evento.x - offset_x, evento.y - offset_y
        if not (0 <= x_display < display_w and 0 <= y_display < display_h):
            return
        cx = int(x_display * w / display_w)
        cy = int(y_display * h / display_h)
        raio = max(1, int(self.s_raio.get() * min(w / display_w, h / display_h)))
        suporte = 3 * raio if self.var_suave.get() else raio
        x0, x1 = max(0, cx - suporte), min(w, cx + suporte + 1)
        y0, y1 = max(0, cy - suporte), min(h, cy + suporte + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        distancia = np.hypot(xx - cx, yy - cy)
        if self.var_suave.get():
            peso = np.exp(-2.0 * (distancia / raio) ** 2)
            peso[distancia > suporte] = 0
        else:
            peso = (distancia <= raio).astype(np.float64)
        nivel = float(self.var_pincel.get())
        self._pintar_area(x0, x1, y0, y1, peso, nivel)

        # Real-valued images require conjugate-symmetric Fourier masks.
        centro_y, centro_x = h // 2, w // 2
        cx_sim, cy_sim = (2 * centro_x - cx) % w, (2 * centro_y - cy) % h
        sx0, sx1 = max(0, cx_sim - suporte), min(w, cx_sim + suporte + 1)
        sy0, sy1 = max(0, cy_sim - suporte), min(h, cy_sim + suporte + 1)
        sy, sx = np.ogrid[sy0:sy1, sx0:sx1]
        distancia_sim = np.hypot(sx - cx_sim, sy - cy_sim)
        if self.var_suave.get():
            peso_sim = np.exp(-2.0 * (distancia_sim / raio) ** 2)
            peso_sim[distancia_sim > suporte] = 0
        else:
            peso_sim = (distancia_sim <= raio).astype(np.float64)
        self._pintar_area(sx0, sx1, sy0, sy1, peso_sim, nivel)
        self._desenhar_espectro()

    def _pintar_area(self, x0, x1, y0, y1, peso, nivel):
        area = self._mascara[y0:y1, x0:x1]
        area[:] = area * (1 - peso) + nivel * peso

    def criar_mascara(self):
        if (self._espectro is None or self._imagem_fonte is not self.app.atual
                or self._metodo_espectro != self.var_metodo.get()):
            self.calcular()
            if self._espectro is None:
                return
        try:
            tipo = self.var_filtro.get()
            centros = [(self.s_notch_x.get(), self.s_notch_y.get())]
            self._mascara = fq.filtro_ideal(
                self._espectro.shape, tipo, self.s_corte.get(),
                self.s_interno.get(), self.s_externo.get(),
                centros=centros, raio_notch=self.s_notch_r.get(),
            )
            self._desenhar_espectro()
        except ValueError as erro:
            messagebox.showerror("Filtro ideal", str(erro), parent=self.app.root)

    def redefinir_mascara(self):
        if self._espectro is not None:
            self._mascara = np.ones(self._espectro.shape, dtype=np.float64)
            self._desenhar_espectro()

    def aplicar_filtro(self):
        if not self.app.precisa_imagem():
            return
        if (self._espectro is None or self._imagem_fonte is not self.app.atual
                or self._metodo_espectro != self.var_metodo.get()):
            self.calcular()
        if self._espectro is None:
            return
        resultado = fq.reconstruir(
            fq.filtrar_espectro(self._espectro, self._mascara),
            self.var_metodo.get(),
            self.var_saida.get(),
        )
        self.app.aplicar_resultado(cinza_para_rgb(resultado))
        self.app.status.set("Filtro de frequência aplicado pela transformada inversa.")

    def aplicar_teorema(self):
        if not self.app.precisa_imagem():
            return
        g = cinza(self.app.atual).astype(np.float64)
        k = kernel_media(3)
        k /= k.sum()
        try:
            filtrada = fq.convolucao_circular_frequencia(g, k, self.var_metodo.get())
        except ValueError as erro:
            messagebox.showerror("Teorema da convolução", str(erro), parent=self.app.root)
            return
        filtrada = np.clip(np.floor(filtrada + 0.5), 0, 255).astype(np.uint8)
        self.app.aplicar_resultado(cinza_para_rgb(filtrada))
        self.app.status.set("Convolução circular calculada pelo teorema da convolução.")

    def aplicar_aliasing(self):
        if not self.app.precisa_imagem():
            return
        resultado = fq.demo_aliasing(
            self.app.atual, int(self.s_fator.get()), self.var_antialias.get())
        self.app.aplicar_resultado(resultado)
        self.app.status.set("Demonstração de subamostragem/aliasing aplicada.")

    def aplicar_moire(self):
        padrao = fq.padrao_moire()
        self.app.aplicar_resultado(cinza_para_rgb(padrao))
        self.app.status.set("Padrão de Moiré sintético gerado.")
