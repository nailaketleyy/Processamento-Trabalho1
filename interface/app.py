import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk   # Pillow: SOMENTE abrir/salvar arquivos e exibir
from algoritmos.base import ajustar_para_tela
from .widgets import espacamento_botao

# Janela principal: menu, abrir/salvar, histórico (desfazer) e exibição.

class App:
    MAX_HIST = 20

    def __init__(self, root, abas):
        self.root = root
        estilo = ttk.Style(root)
        estilo.configure("TButton", padding=(10, 6))
        estilo.configure("TNotebook.Tab", padding=(8, 6))
        root.bind_class("TButton", "<Map>", espacamento_botao, add="+")
        root.title("Processador de Imagens")
        root.geometry("1280x760")
        root.minsize(900, 600)

        self.original = None   # imagem como foi aberta
        self.atual = None      # imagem após as operações aplicadas
        self.previa = None     # pré-visualização (ainda não aplicada)
        self.historico = []
        self.tem_esteg = False
        self.observadores = []  # funções chamadas quando a exibição muda
        self._foto_orig = None
        self._foto_res = None

        self.status = tk.StringVar(value="Abra uma imagem (Arquivo > Abrir) para começar.")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

        self._menu()
        self._layout()
        for classe in abas:
            classe(self, self.notebook)

    # ---------------- estrutura ----------------
    def _menu(self):
        m = tk.Menu(self.root)
        arq = tk.Menu(m, tearoff=0)
        arq.add_command(label="Abrir...", accelerator="Ctrl+O", command=self.abrir)
        arq.add_command(label="Salvar como...", accelerator="Ctrl+S", command=self.salvar)
        arq.add_separator()
        arq.add_command(label="Sair", command=self.root.destroy)
        m.add_cascade(label="Arquivo", menu=arq)
        ed = tk.Menu(m, tearoff=0)
        ed.add_command(label="Desfazer", accelerator="Ctrl+Z", command=self.desfazer)
        ed.add_command(label="Restaurar original", command=self.restaurar)
        m.add_cascade(label="Editar", menu=ed)
        self.root.config(menu=m)
        self.root.bind("<Control-o>", lambda e: self.abrir())
        self.root.bind("<Control-s>", lambda e: self.salvar())
        self.root.bind("<Control-z>", lambda e: self.desfazer())

    def _layout(self):
        esquerda = ttk.Frame(self.root, width=400)
        esquerda.pack(side="left", fill="y", padx=6, pady=6)
        esquerda.pack_propagate(False)

        self.notebook = ttk.Notebook(esquerda)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._ao_trocar_aba())

        barra = ttk.Frame(esquerda)
        barra.pack(fill="x", pady=(6, 0))
        ttk.Button(barra, text="Desfazer", command=self.desfazer).pack(side="left", expand=True, fill="x")
        ttk.Button(barra, text="Restaurar original", command=self.restaurar).pack(
            side="left", expand=True, fill="x", padx=(4, 0))

        direita = ttk.Frame(self.root)
        direita.pack(side="right", fill="both", expand=True, padx=(0, 6), pady=6)
        direita.columnconfigure((0, 1), weight=1)
        direita.rowconfigure(1, weight=1)
        ttk.Label(direita, text="Original", font=("", 11, "bold")).grid(row=0, column=0)
        ttk.Label(direita, text="Resultado (pré-visualização antes de aplicar)",
                  font=("", 11, "bold")).grid(row=0, column=1)
        self.cv_orig = tk.Canvas(direita, bg="#2b2b2b", highlightthickness=0)
        self.cv_res = tk.Canvas(direita, bg="#2b2b2b", highlightthickness=0)
        self.cv_orig.grid(row=1, column=0, sticky="nsew", padx=(0, 3))
        self.cv_res.grid(row=1, column=1, sticky="nsew", padx=(3, 0))
        self.cv_orig.bind("<Configure>", lambda e: self.atualizar())
        self.cv_res.bind("<Configure>", lambda e: self.atualizar())

    # ---------------- arquivos / histórico ----------------
    def abrir(self):
        p = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.bmp *.jpg *.jpeg *.png"), ("Todos", "*.*")])
        if not p:
            return
        try:
            img = np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível abrir a imagem:\n{e}")
            return
        self.original = img
        self.atual = img.copy()
        self.historico = []
        self.previa = None
        self.tem_esteg = False
        self.status.set(f"{p}  -  {img.shape[1]} x {img.shape[0]} px")
        self.atualizar()

    def salvar(self):
        if not self.precisa_imagem():
            return
        p = filedialog.asksaveasfilename(
            defaultextension=".bmp",
            filetypes=[("Bitmap", "*.bmp"), ("JPEG", "*.jpg *.jpeg"), ("PNG", "*.png")])
        if not p:
            return
        if self.tem_esteg and p.lower().endswith((".jpg", ".jpeg")):
            if not messagebox.askyesno(
                    "Aviso",
                    "JPG usa compressão com perdas e DESTRUIRÁ a mensagem oculta.\nSalvar mesmo assim?"):
                return
        try:
            img = Image.fromarray(self.atual)
            if p.lower().endswith((".jpg", ".jpeg")):
                img.save(p, quality=95)
            else:
                img.save(p)
            self.status.set(f"Salvo em {p}")
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível salvar:\n{e}")

    def precisa_imagem(self):
        if self.atual is None:
            messagebox.showinfo("Aviso", "Abra uma imagem primeiro.")
            return False
        return True

    # ---------------- API usada pelas abas ----------------
    def aplicar(self, fn):
        """Executa fn() (que devolve a nova matriz) e aplica à imagem."""
        if not self.precisa_imagem():
            return
        self.root.config(cursor="watch")
        self.status.set("Processando...")
        self.root.update_idletasks()
        try:
            nova = fn()
        finally:
            self.root.config(cursor="")
        self.aplicar_resultado(nova)
        self.status.set("Pronto.")

    def aplicar_resultado(self, nova):
        self.historico.append(self.atual)
        if len(self.historico) > self.MAX_HIST:
            self.historico.pop(0)
        self.atual = nova
        self.previa = None
        self.atualizar()

    def prever(self, fn):
        """Mostra fn() no painel de resultado sem alterar a imagem."""
        if self.atual is None:
            return
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            self.previa = fn()
        finally:
            self.root.config(cursor="")
        self._desenhar(self.cv_res, self.previa, "_foto_res")
        self._notificar()

    def desfazer(self):
        if self.historico:
            self.atual = self.historico.pop()
            self.previa = None
            self.atualizar()

    def restaurar(self):
        if self.original is not None:
            self.historico.append(self.atual)
            self.atual = self.original.copy()
            self.previa = None
            self.tem_esteg = False
            self.atualizar()

    # ---------------- exibição ----------------
    def _ao_trocar_aba(self):
        self.previa = None
        self.atualizar()

    def ao_atualizar(self, callback):
        """Registra uma função (sem argumentos) chamada sempre que a exibição muda."""
        self.observadores.append(callback)

    def _notificar(self):
        for cb in self.observadores:
            cb()

    def resultado_atual(self):
        """Matriz exibida no painel de resultado (pré-visualização ou imagem atual)."""
        return self.previa if self.previa is not None else self.atual

    def atualizar(self):
        self._desenhar(self.cv_orig, self.original, "_foto_orig")
        res = self.previa if self.previa is not None else self.atual
        self._desenhar(self.cv_res, res, "_foto_res")
        self._notificar()

    def _desenhar(self, canvas, img, attr):
        canvas.delete("all")
        if img is None:
            return
        w, h = max(canvas.winfo_width(), 10), max(canvas.winfo_height(), 10)
        foto = ImageTk.PhotoImage(Image.fromarray(ajustar_para_tela(img, w - 4, h - 4)))
        canvas.create_image(w // 2, h // 2, image=foto)
        setattr(self, attr, foto)  # mantém referência
