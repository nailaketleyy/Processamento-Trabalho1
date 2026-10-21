"""ETAPA 1 - Aba: Esteganografia."""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from algoritmos import esteganografia as es

from .aba_base import Aba


class AbaEsteganografia(Aba):
    titulo = "Esteganografia"

    def construir(self, f):
        lf = ttk.LabelFrame(f, text="Ocultar")
        lf.pack(fill="x", padx=6, pady=4)
        linha = ttk.Frame(lf)
        linha.pack(pady=4)
        ttk.Label(linha, text="Bits por canal (LSB):").pack(side="left")
        self.k_var = tk.IntVar(value=1)
        ttk.Spinbox(linha, from_=1, to=4, width=4, textvariable=self.k_var).pack(side="left", padx=4)
        ttk.Button(lf, text="Ocultar texto...", command=self.dlg_ocultar_texto).pack(fill="x", padx=6, pady=2)
        ttk.Button(lf, text="Ocultar arquivo qualquer...", command=self.ocultar_arquivo).pack(
            fill="x", padx=6, pady=(2, 6))

        lf = ttk.LabelFrame(f, text="Extrair")
        lf.pack(fill="x", padx=6, pady=4)
        ttk.Button(lf, text="Extrair mensagem da imagem...", command=self.dlg_extrair).pack(
            fill="x", padx=6, pady=6)

        ttk.Label(f, wraplength=290, foreground="#a51d2d", justify="left",
                  text="Atenção: salve a imagem em BMP ou PNG (sem perdas). "
                       "O formato JPG altera os pixels e destrói a mensagem oculta. "
                       "Mais bits por canal = mais capacidade, porém mais distorção visível."
                  ).pack(padx=8, pady=8)

    def _k(self):
        try:
            return min(4, max(1, int(self.k_var.get())))
        except (tk.TclError, ValueError):
            return 1

    def dlg_ocultar_texto(self):
        app = self.app
        if not app.precisa_imagem():
            return
        k = self._k()
        cap = es.capacidade(app.atual, k)
        win = tk.Toplevel(app.root)
        win.title("Ocultar texto")
        ttk.Label(win, text=f"Capacidade: {cap} bytes ({k} bit(s) por canal)").pack(padx=8, pady=4)
        txt = tk.Text(win, width=60, height=14)
        txt.pack(padx=8, pady=4)

        def carregar():
            p = filedialog.askopenfilename(filetypes=[("Texto", "*.txt"), ("Todos", "*.*")])
            if p:
                with open(p, "r", encoding="utf-8", errors="replace") as fh:
                    txt.delete("1.0", "end")
                    txt.insert("1.0", fh.read())

        def ocultar():
            dados = txt.get("1.0", "end-1c").encode("utf-8")
            if self._ocultar_bytes(dados, k):
                win.destroy()

        bt = ttk.Frame(win)
        bt.pack(pady=6)
        ttk.Button(bt, text="Carregar de .txt...", command=carregar).pack(side="left", padx=4)
        ttk.Button(bt, text="Ocultar na imagem", command=ocultar).pack(side="left", padx=4)

    def ocultar_arquivo(self):
        if not self.app.precisa_imagem():
            return
        p = filedialog.askopenfilename(title="Arquivo a ser ocultado")
        if p:
            with open(p, "rb") as fh:
                self._ocultar_bytes(fh.read(), self._k())

    def _ocultar_bytes(self, dados, k):
        app = self.app
        try:
            nova = es.ocultar(app.atual, dados, k)
        except ValueError as e:
            messagebox.showerror("Esteganografia", str(e))
            return False
        app.aplicar_resultado(nova)
        app.tem_esteg = True
        app.status.set(f"{len(dados)} bytes ocultados. Salve em BMP ou PNG para preservar a mensagem.")
        return True

    def dlg_extrair(self):
        app = self.app
        if not app.precisa_imagem():
            return
        try:
            dados = es.extrair(app.atual)
        except ValueError as e:
            messagebox.showinfo("Esteganografia", str(e))
            return
        win = tk.Toplevel(app.root)
        win.title("Mensagem extraída")
        ttk.Label(win, text=f"{len(dados)} bytes extraídos").pack(padx=8, pady=4)
        txt = tk.Text(win, width=60, height=14)
        txt.pack(padx=8, pady=4)
        txt.insert("1.0", dados.decode("utf-8", errors="replace"))

        def salvar_bin():
            p = filedialog.asksaveasfilename(title="Salvar dado extraído")
            if p:
                with open(p, "wb") as fh:
                    fh.write(dados)

        ttk.Button(win, text="Salvar como arquivo...", command=salvar_bin).pack(pady=6)
