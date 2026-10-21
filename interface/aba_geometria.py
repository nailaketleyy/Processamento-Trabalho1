"""Aba de transformações geométricas: escala e rotação com pivô."""

import tkinter as tk
from tkinter import ttk, messagebox

from algoritmos.geometria import escalar, rotacionar

from .aba_base import Aba
from .widgets import slider


class AbaGeometria(Aba):
    titulo = "Geometria"

    def construir(self, f):
        ttk.Label(
            f,
            text=("Mapeamento inverso. A escala altera as dimensões da imagem; a rotação "
                  "expande o canvas para não cortar. Regiões sem origem ficam pretas."),
            wraplength=400, justify="left",
        ).pack(fill="x", padx=8, pady=8)

        lf = ttk.LabelFrame(f, text="Escala (ampliação / redução)")
        lf.pack(fill="x", padx=6, pady=4)
        self.sx = slider(lf, "Fator horizontal sx", 0.1, 4.0, 1.0, 0.05)
        self.sy = slider(lf, "Fator vertical sy", 0.1, 4.0, 1.0, 0.05)
        ttk.Button(lf, text="Aplicar escala", command=self.aplicar_escala).pack(pady=3)

        lf = ttk.LabelFrame(f, text="Rotação em torno de um pivô")
        lf.pack(fill="x", padx=6, pady=4)
        self.angulo = slider(lf, "Ângulo em graus (positivo = horário)", -180, 180, 0, 1)
        self.var_centro = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Usar centro da imagem como pivô",
                        variable=self.var_centro, command=self._alternar_pivo,
                        ).pack(anchor="w", padx=6)
        linha = ttk.Frame(lf)
        linha.pack(fill="x", padx=6, pady=4)
        ttk.Label(linha, text="x:").pack(side="left")
        self.pivo_x = tk.StringVar(value="0")
        self.ent_x = ttk.Entry(linha, textvariable=self.pivo_x, width=7)
        self.ent_x.pack(side="left", padx=(2, 8))
        ttk.Label(linha, text="y:").pack(side="left")
        self.pivo_y = tk.StringVar(value="0")
        self.ent_y = ttk.Entry(linha, textvariable=self.pivo_y, width=7)
        self.ent_y.pack(side="left", padx=2)
        ttk.Label(lf, text="Também é possível clicar na imagem de resultado para escolher o pivô.",
                  wraplength=400).pack(anchor="w", padx=6)
        self.var_expandir = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf, text="Expandir canvas (não cortar a imagem)",
                        variable=self.var_expandir).pack(anchor="w", padx=6)
        ttk.Button(lf, text="Aplicar rotação", command=self.aplicar_rotacao).pack(pady=3)
        self._alternar_pivo()
        self.app.cv_res.bind("<Button-1>", self._clicou, add="+")

        lf = ttk.LabelFrame(f, text="Interpolação")
        lf.pack(fill="x", padx=6, pady=4)
        self.var_interpolacao = tk.StringVar(value="vizinho")
        ttk.Radiobutton(lf, text="Vizinho mais próximo", value="vizinho",
                        variable=self.var_interpolacao).pack(anchor="w", padx=6, pady=2)
        ttk.Radiobutton(lf, text="Bilinear", value="bilinear",
                        variable=self.var_interpolacao).pack(anchor="w", padx=6, pady=2)

        ttk.Button(f, text="Pré-visualizar escala", command=lambda: self._executar(self.fn_escala, True)
                   ).pack(fill="x", padx=8, pady=2)
        ttk.Button(f, text="Pré-visualizar rotação", command=lambda: self._executar(self.fn_rotacao, True)
                   ).pack(fill="x", padx=8, pady=2)

    def _alternar_pivo(self):
        estado = "disabled" if self.var_centro.get() else "normal"
        self.ent_x.configure(state=estado)
        self.ent_y.configure(state=estado)

    def _clicou(self, evento):
        if not self.visivel() or self.app.atual is None:
            return
        if self.app.previa is not None and self.app.previa.shape != self.app.atual.shape:
            self.app.status.set("Aplique ou descarte a prévia antes de escolher o pivô.")
            return
        pos = self.app.coordenada_na_imagem(self.app.cv_res, evento)
        if pos is None:
            return
        self.var_centro.set(False)
        self._alternar_pivo()
        self.pivo_x.set(str(pos[0]))
        self.pivo_y.set(str(pos[1]))
        self.app.status.set(f"Pivô definido em x={pos[0]}, y={pos[1]}.")

    def fn_escala(self):
        return escalar(self.app.atual, self.sx.get(), self.sy.get(),
                       self.var_interpolacao.get())

    def fn_rotacao(self):
        if self.var_centro.get():
            pivo = None
        else:
            try:
                pivo = (float(self.pivo_x.get().replace(",", ".")),
                        float(self.pivo_y.get().replace(",", ".")))
            except ValueError:
                raise ValueError("O pivô deve conter números.")
        return rotacionar(self.app.atual, self.angulo.get(), pivo,
                          self.var_interpolacao.get(), self.var_expandir.get())

    def _executar(self, fn, previsualizar):
        if not self.app.precisa_imagem():
            return
        try:
            (self.app.prever if previsualizar else self.app.aplicar)(fn)
        except ValueError as erro:
            messagebox.showerror("Transformação geométrica", str(erro), parent=self.app.root)

    def aplicar_escala(self):
        self._executar(self.fn_escala, False)

    def aplicar_rotacao(self):
        self._executar(self.fn_rotacao, False)
