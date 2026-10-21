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
           "passa-notch", "rejeita-notch",
           "butterworth passa-baixa", "butterworth passa-alta",
           "gaussiano passa-baixa", "gaussiano passa-alta")


class AbaFrequencia(Aba):
    titulo = "Frequência / Aliasing"

    def construir(self, f):
        self._imagem_fonte = None
        self._espectro = None
        self._mascara = None
        self._metodo_espectro = None
        self._foto_espectro = None
        self._base_visual = None
        self._desenho_agendado = None
        self._display_shape = (0, 0)
        self._display_offset = (0, 0)
        ttk.Label(
            f,
            text=("Pipeline: zero padding (P≥2M−1, Q≥2N−1) → transformada → H·F → inversa → "
                "recorte. A máscara tem o tamanho preenchido P×Q. DFT direta: até 1024 "
                "por dimensão; FFT própria: padding vira potência de 2; 'fft' usa NumPy."),
            wraplength=400, justify="left",
        ).pack(fill="x", padx=8, pady=6)

        lf = ttk.LabelFrame(f, text="Espectro e filtro ideal")
        lf.pack(fill="x", padx=6, pady=4)
        linha = ttk.Frame(lf)
        linha.pack(fill="x", padx=5, pady=3)
        ttk.Label(linha, text="Transformada:").pack(side="left")
        self.var_metodo = tk.StringVar(value="fft")
        ttk.Combobox(
            linha, textvariable=self.var_metodo,
            values=("fft", "fft-propria", "dft"),
            state="readonly", width=12,
        ).pack(side="left", padx=8)
        ttk.Button(lf, text="Calcular espectro", command=self.calcular).pack(
            fill="x", padx=8, pady=6)
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
        self.s_ordem = slider(lf, "Ordem n (Butterworth)", 1, 8, 2, 1)
        self.s_corte = slider(lf, "Corte (baixa/alta)", 0, 512, 24, 1)
        self.s_interno = slider(lf, "Raio interno (faixa)", 0, 256, 10, 1)
        self.s_externo = slider(lf, "Raio externo (faixa)", 0, 256, 40, 1)
        self.s_notch_x = slider(lf, "Centro notch u (deslocamento vertical)", -128, 128, 30, 1)
        self.s_notch_y = slider(lf, "Centro notch v (deslocamento horizontal)", -128, 128, 30, 1)
        self.s_notch_r = slider(lf, "Raio do notch", 0, 64, 5, 1)
        self._centros_notch = []
        ttk.Label(lf, text="Centros notch (cada centro também remove o simétrico):").pack(
            anchor="w", padx=5)
        self.lb_centros_notch = tk.Listbox(lf, height=3, exportselection=False)
        self.lb_centros_notch.pack(fill="x", padx=5, pady=2)
        linha_notch = ttk.Frame(lf)
        linha_notch.pack(fill="x")
        linha_notch.columnconfigure((0, 1), weight=1)
        ttk.Button(
            linha_notch, text="Adicionar centro",
            command=self._adicionar_centro_notch,
        ).grid(row=0, column=0, sticky="ew", padx=(8, 4), pady=3)
        ttk.Button(
            linha_notch, text="Remover selecionado",
            command=self._remover_centro_notch,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 8), pady=3)
        self._adicionar_centro_notch()
        ttk.Button(lf, text="Criar máscara do filtro", command=self.criar_mascara).pack(
            fill="x", padx=8, pady=6)

        lf = ttk.LabelFrame(f, text="Editar o espectro deslocado")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            lf, text="Arraste sobre o espectro. Preto rejeita; branco mantém; "
                     "cinza atenua parcialmente.",
            wraplength=400, justify="left",
        ).pack(anchor="w", padx=5, pady=3)
        self.cv_espectro = tk.Canvas(
            lf, width=400, height=280, background="black", highlightthickness=1)
        self.cv_espectro.pack(padx=4, pady=3)
        self.cv_espectro.bind("<Button-1>", self._pintar_evento)
        self.cv_espectro.bind("<B1-Motion>", self._pintar_evento)
        linha = ttk.Frame(lf)
        linha.pack(fill="x")
        self.var_pincel = tk.DoubleVar(value=0.0)
        ttk.Label(linha, text="Nível do pincel (0 = preto, 1 = branco):").pack(
            side="left", padx=8)
        ttk.Combobox(
            linha, textvariable=self.var_pincel,
            values=(0.0, 0.25, 0.5, 0.75, 1.0), state="readonly", width=5,
        ).pack(side="left", padx=4)
        self.var_suave = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Pincel suave", variable=self.var_suave).pack(
            anchor="w", padx=8, pady=2)
        self.s_raio = slider(lf, "Raio do pincel (pixels no espectro)", 1, 50, 12, 1)
        botoes = ttk.Frame(lf)
        botoes.pack(fill="x", pady=3)
        ttk.Button(botoes, text="Redefinir máscara", command=self.redefinir_mascara).pack(
            fill="x", padx=8, pady=3)
        ttk.Button(botoes, text="Aplicar filtro / IDFT", command=self.aplicar_filtro).pack(
            fill="x", padx=8, pady=3)

        lf = ttk.LabelFrame(f, text="Teorema da convolução")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Label(
            lf, text="Demonstra convolução circular imagem × kernel no domínio "
                     "da frequência, usando um kernel média 3×3.",
            wraplength=400, justify="left",
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
            self._base_visual = None
            self._espectro = None
            self._mascara = None
            self._metodo_espectro = None
            self.cv_espectro.delete("all")
            self.cv_espectro.create_text(
                200, 140, text="Calcule o espectro após abrir uma imagem",
                fill="white", width=360,
            )

    def calcular(self):
        if not self.app.precisa_imagem():
            return
        try:
            g = cinza(self.app.atual)
            metodo = self.var_metodo.get()
            self.app.root.config(cursor="watch")
            self.app.root.update_idletasks()
            forma = fq.forma_preenchida(g.shape, metodo)
            espectro = fq.transformar(fq.preencher_zeros(g, forma), metodo)
            self._espectro = espectro
            self._base_visual = None
            self._mascara = np.ones(forma, dtype=np.float64)
            self._metodo_espectro = metodo
            self._imagem_fonte = self.app.atual
            self._desenhar_espectro()
            self.app.status.set(f"Espectro {forma[1]}×{forma[0]} calculado (com zero padding).")
        except ValueError as erro:
            self._espectro = None
            self._mascara = None
            self._metodo_espectro = None
            messagebox.showerror("Transformada de Fourier", str(erro), parent=self.app.root)
        finally:
            self.app.root.config(cursor="")

    LARG_CV, ALT_CV = 400, 280

    def _preparar_exibicao(self):
        """Calcula UMA vez o espectro (log) já reduzido ao tamanho do canvas
        e os índices de amostragem; o pincel só atualiza essa miniatura."""
        h, w = self._espectro.shape
        escala = min(self.LARG_CV / w, self.ALT_CV / h)
        dw, dh = max(1, int(round(w * escala))), max(1, int(round(h * escala)))
        self._display_shape = (dh, dw)
        self._display_offset = ((self.LARG_CV - dw) // 2, (self.ALT_CV - dh) // 2)
        self._idx_y = np.minimum((np.arange(dh) * h / dh).astype(int), h - 1)
        self._idx_x = np.minimum((np.arange(dw) * w / dw).astype(int), w - 1)
        base = fq.espectro_exibicao(self._espectro)          # custo pago só aqui
        self._base_visual = base[self._idx_y][:, self._idx_x].astype(np.float64)

    def _desenhar_espectro(self):
        if self._espectro is None:
            return
        if self._base_visual is None:
            self._preparar_exibicao()
        mascara = self._mascara[self._idx_y][:, self._idx_x]
        visual = np.floor(self._base_visual * mascara + 0.5).astype(np.uint8)
        self._foto_espectro = ImageTk.PhotoImage(Image.fromarray(visual))
        self.cv_espectro.delete("all")
        self.cv_espectro.create_image(
            self._display_offset[0], self._display_offset[1],
            image=self._foto_espectro, anchor="nw")

    def _agendar_desenho(self):
        """Junta vários eventos de mouse em um único redesenho (~60 fps)."""
        if self._desenho_agendado is None:
            self._desenho_agendado = self.cv_espectro.after(16, self._redesenhar)

    def _redesenhar(self):
        self._desenho_agendado = None
        self._desenhar_espectro()

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
        self._agendar_desenho()

    def _pintar_area(self, x0, x1, y0, y1, peso, nivel):
        area = self._mascara[y0:y1, x0:x1]
        area[:] = area * (1 - peso) + nivel * peso

    def _adicionar_centro_notch(self):
        centro = (int(self.s_notch_x.get()), int(self.s_notch_y.get()))
        if centro not in self._centros_notch:
            self._centros_notch.append(centro)
            self.lb_centros_notch.insert("end", f"u={centro[0]}, v={centro[1]}")

    def _remover_centro_notch(self):
        selecionados = self.lb_centros_notch.curselection()
        if selecionados:
            indice = selecionados[0]
            self.lb_centros_notch.delete(indice)
            del self._centros_notch[indice]

    def criar_mascara(self):
        if (self._espectro is None or self._imagem_fonte is not self.app.atual
                or self._metodo_espectro != self.var_metodo.get()):
            self.calcular()
            if self._espectro is None:
                return
        try:
            tipo = self.var_filtro.get()
            if tipo.startswith(("butterworth", "gaussiano")):
                familia, base = tipo.split(" ", 1)
                self._mascara = fq.filtro_suave(
                    self._espectro.shape, familia, base,
                    max(1.0, float(self.s_corte.get())), 10.0, int(self.s_ordem.get()))
                self._desenhar_espectro()
                return
            self._mascara = fq.filtro_ideal(
                self._espectro.shape, tipo, self.s_corte.get(),
                self.s_interno.get(), self.s_externo.get(),
                centros=self._centros_notch,
                raio_notch=self.s_notch_r.get(),
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
        self.app.root.config(cursor="watch")
        self.app.root.update_idletasks()
        try:
            resultado = fq.filtrar_rgb(
                self.app.atual, self._mascara,
                self.var_metodo.get(), self.var_saida.get(),
            )
        except ValueError as erro:
            messagebox.showerror("Filtro de frequência", str(erro), parent=self.app.root)
            return
        finally:
            self.app.root.config(cursor="")
        self.app.aplicar_resultado(resultado)
        self.app.status.set("Filtro de frequência aplicado pela transformada inversa.")

    def aplicar_teorema(self):
        if not self.app.precisa_imagem():
            return
        k = kernel_media(3)
        k /= k.sum()
        try:
            filtrada = fq.convolucao_circular_rgb(
                self.app.atual, k, self.var_metodo.get())
        except ValueError as erro:
            messagebox.showerror("Teorema da convolução", str(erro), parent=self.app.root)
            return
        filtrada = np.clip(np.floor(filtrada + 0.5), 0, 255).astype(np.uint8)
        self.app.aplicar_resultado(filtrada)
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
