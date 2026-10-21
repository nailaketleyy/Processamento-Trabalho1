import tkinter as tk
import tkinter.font as tkfont
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
        for nome in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(nome).configure(size=10)
        estilo = ttk.Style(root)
        estilo.configure("TButton", padding=(12, 7))
        estilo.configure("TLabelframe", padding=(8, 6))
        estilo.configure("TLabelframe.Label", font=("", 10, "bold"))
        estilo.configure("TCheckbutton", padding=(2, 3))
        estilo.configure("TRadiobutton", padding=(2, 3))
        estilo.configure("TCombobox", padding=(4, 4))
        estilo.layout("TNotebook.Tab", [])     # abas escondidas: a navegação é o seletor de ferramentas
        root.bind_class("TButton", "<Map>", espacamento_botao, add="+")
        root.title("Processador de Imagens")
        root.geometry("1380x800")
        root.minsize(900, 600)

        self.original = None   # imagem como foi aberta
        self.atual = None      # imagem após as operações aplicadas
        self.previa = None     # pré-visualização (ainda não aplicada)
        self.historico = []
        self.tem_esteg = False
        self.observadores = []  # funções chamadas quando a exibição muda
        self._exibido = {}      # canvas -> (matriz exibida, largura, altura e origem na tela)
        self._foto_orig = None
        self._foto_res = None

        self.status = tk.StringVar(value="Abra uma imagem (Arquivo > Abrir) para começar.")
        ttk.Label(root, textvariable=self.status, relief="sunken", anchor="w").pack(
            side="bottom", fill="x")

        self._menu()
        self._layout()
        for classe in abas:
            classe(self, self.notebook)
        nomes = [self.notebook.tab(i, "text") for i in range(self.notebook.index("end"))]
        self.cb_ferramenta.configure(values=nomes, height=min(14, len(nomes)))
        if nomes:
            self.var_ferramenta.set(nomes[0])

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
        esquerda = ttk.Frame(self.root, width=460)
        esquerda.pack(side="left", fill="y", padx=6, pady=6)
        esquerda.pack_propagate(False)

        seletor = ttk.Frame(esquerda)
        seletor.pack(fill="x", pady=(0, 8))
        ttk.Label(seletor, text="Ferramenta:", font=("", 10, "bold")).pack(side="left", padx=(0, 8))
        self.var_ferramenta = tk.StringVar()
        self.cb_ferramenta = ttk.Combobox(seletor, textvariable=self.var_ferramenta,
                                          state="readonly", font=("", 11))
        self.cb_ferramenta.pack(side="left", fill="x", expand=True)
        self.cb_ferramenta.bind("<<ComboboxSelected>>", lambda e: self._escolher_ferramenta())

        self.notebook = ttk.Notebook(esquerda)
        self.notebook.pack(fill="both", expand=True)
        self.notebook.bind("<<NotebookTabChanged>>", lambda e: self._ao_trocar_aba())

        barra = ttk.Frame(esquerda)
        barra.pack(fill="x", pady=(8, 0))
        barra.columnconfigure((0, 1), weight=1)
        ttk.Button(barra, text="Desfazer", command=self.desfazer).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=3)
        ttk.Button(barra, text="Restaurar original", command=self.restaurar).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=3)
        ttk.Button(barra, text="Usar resultado como entrada", command=self.usar_previa).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=3)

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

    def usar_previa(self):
        """Torna a pré-visualização atual a nova imagem de entrada (entra no histórico)."""
        if self.previa is None:
            self.status.set("Não há pré-visualização para usar como entrada.")
            return
        self.aplicar_resultado(self.previa)
        self.status.set("Resultado usado como nova entrada.")

    def coordenada_na_imagem(self, canvas, evento):
        """Converte um clique no canvas em (x, y) da matriz exibida (base 0).
        Devolve None se o clique cair fora da imagem."""
        info = self._exibido.get(canvas)
        if info is None:
            return None
        img, esq, topo, nw, nh = info
        dx, dy = evento.x - esq, evento.y - topo
        if not (0 <= dx < nw and 0 <= dy < nh):
            return None
        h, w = img.shape[:2]
        return min(w - 1, int(dx * w / nw)), min(h - 1, int(dy * h / nh))

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
    def _escolher_ferramenta(self):
        nomes = self.cb_ferramenta.cget("values")
        if isinstance(nomes, str):
            nomes = self.cb_ferramenta.tk.splitlist(nomes)
        self.notebook.select(list(nomes).index(self.var_ferramenta.get()))

    def _ao_trocar_aba(self):
        try:
            self.var_ferramenta.set(self.notebook.tab(self.notebook.select(), "text"))
        except tk.TclError:
            pass
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
        self._exibido.pop(canvas, None)
        if img is None:
            return
        w, h = max(canvas.winfo_width(), 10), max(canvas.winfo_height(), 10)
        reduzida = ajustar_para_tela(img, w - 4, h - 4)
        foto = ImageTk.PhotoImage(Image.fromarray(reduzida))
        canvas.create_image(w // 2, h // 2, image=foto)
        nh, nw = reduzida.shape[:2]
        self._exibido[canvas] = (img, w // 2 - nw // 2, h // 2 - nh // 2, nw, nh)
        setattr(self, attr, foto)  # mantém referência
