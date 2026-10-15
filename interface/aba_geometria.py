"""Aba de transformações geométricas da imagem."""

import tkinter as tk
from tkinter import ttk, messagebox

from algoritmos.geometria import transformar

from .aba_base import Aba
from .widgets import slider


class AbaGeometria(Aba):
    titulo = "Geometria"

    def construir(self, f):
        ttk.Label(
            f,
            text=("Escala, rotação e translação são aplicadas por mapeamento "
                  "inverso. A saída mantém o tamanho original; áreas sem pixels "
                  "de origem ficam pretas."),
            wraplength=330,
            justify="left",
        ).pack(fill="x", padx=8, pady=8)

        lf = ttk.LabelFrame(f, text="Escala")
        lf.pack(fill="x", padx=6, pady=4)
        self.sx = slider(lf, "Escala horizontal", 0.1, 3.0, 1.0, 0.1)
        self.sy = slider(lf, "Escala vertical", 0.1, 3.0, 1.0, 0.1)

        lf = ttk.LabelFrame(f, text="Rotação e pivô (coordenadas x, y)")
        lf.pack(fill="x", padx=6, pady=4)
        self.angulo = slider(lf, "Ângulo em graus (positivo = horário)", -180, 180, 0, 1)
        self.var_centro = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            lf, text="Usar centro da imagem como pivô",
            variable=self.var_centro, command=self._alternar_pivo,
        ).pack(anchor="w", padx=6)
        coordenadas = ttk.Frame(lf)
        coordenadas.pack(fill="x", padx=6, pady=4)
        ttk.Label(coordenadas, text="x:").pack(side="left")
        self.pivo_x = tk.StringVar(value="0")
        self.ent_x = ttk.Entry(coordenadas, textvariable=self.pivo_x, width=8)
        self.ent_x.pack(side="left", padx=(2, 8))
        ttk.Label(coordenadas, text="y:").pack(side="left")
        self.pivo_y = tk.StringVar(value="0")
        self.ent_y = ttk.Entry(coordenadas, textvariable=self.pivo_y, width=8)
        self.ent_y.pack(side="left", padx=2)
        self._alternar_pivo()

        lf = ttk.LabelFrame(f, text="Translação (pixels)")
        lf.pack(fill="x", padx=6, pady=4)
        coordenadas = ttk.Frame(lf)
        coordenadas.pack(fill="x", padx=6, pady=6)
        ttk.Label(coordenadas, text="dx:").pack(side="left")
        self.dx = tk.StringVar(value="0")
        ttk.Entry(coordenadas, textvariable=self.dx, width=8).pack(side="left", padx=(2, 8))
        ttk.Label(coordenadas, text="dy:").pack(side="left")
        self.dy = tk.StringVar(value="0")
        ttk.Entry(coordenadas, textvariable=self.dy, width=8).pack(side="left", padx=2)
        ttk.Label(lf, text="dx positivo move para a direita; dy positivo, para baixo.",
                  wraplength=320).pack(anchor="w", padx=6, pady=(0, 6))

        lf = ttk.LabelFrame(f, text="Interpolação")
        lf.pack(fill="x", padx=6, pady=4)
        self.var_interpolacao = tk.StringVar(value="vizinho")
        ttk.Radiobutton(
            lf, text="Vizinho mais próximo", value="vizinho",
            variable=self.var_interpolacao,
        ).pack(anchor="w", padx=6, pady=2)
        ttk.Radiobutton(
            lf, text="Bilinear", value="bilinear",
            variable=self.var_interpolacao,
        ).pack(anchor="w", padx=6, pady=2)

        botoes = ttk.Frame(f)
        botoes.pack(fill="x", padx=8, pady=8)
        ttk.Button(
            botoes, text="Pré-visualizar",
            command=self.pre_visualizar,
        ).pack(fill="x", pady=2)
        ttk.Button(
            botoes, text="Aplicar transformação",
            command=self.aplicar,
        ).pack(fill="x", pady=2)

    def _alternar_pivo(self):
        estado = "disabled" if self.var_centro.get() else "normal"
        self.ent_x.configure(state=estado)
        self.ent_y.configure(state=estado)

    def fn_transformar(self):
        if self.var_centro.get():
            pivo = None
        else:
            pivo = (float(self.pivo_x.get().replace(",", ".")),
                    float(self.pivo_y.get().replace(",", ".")))
        deslocamento = (float(self.dx.get().replace(",", ".")),
                        float(self.dy.get().replace(",", ".")))
        return transformar(
            self.app.atual,
            escala=(self.sx.get(), self.sy.get()),
            angulo=self.angulo.get(),
            deslocamento=deslocamento,
            pivo=pivo,
            interpolacao=self.var_interpolacao.get(),
        )

    def _executar(self, previsualizar):
        try:
            if previsualizar:
                self.app.prever(self.fn_transformar)
            else:
                self.app.aplicar(self.fn_transformar)
        except ValueError as erro:
            messagebox.showerror(
                "Transformação geométrica", str(erro), parent=self.app.root)

    def pre_visualizar(self):
        self._executar(previsualizar=True)

    def aplicar(self):
        self._executar(previsualizar=False)
