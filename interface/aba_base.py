"""Classe base das abas de ferramentas.

Para criar uma aba nova (nova etapa do trabalho):
  1. crie interface/aba_<tema>.py com uma subclasse de Aba;
  2. defina `titulo` e implemente `construir(self, frame)`;
  3. registre a classe na lista ABAS em main.py.

O conteúdo da aba é colocado em `frame`, que já tem barra de rolagem vertical
(abas com muitos controles continuam acessíveis em telas pequenas).

Dentro da aba use a API do aplicativo (self.app):
  self.app.atual                  matriz da imagem atual (None se não houver)
  self.app.original               matriz da imagem como foi aberta
  self.app.precisa_imagem()       avisa o usuário e devolve False se não há imagem
  self.app.prever(fn)             mostra fn() como pré-visualização (sem aplicar)
  self.app.aplicar(fn)            aplica fn() à imagem (entra no histórico)
  self.app.aplicar_resultado(m)   aplica uma matriz já calculada
  self.app.resultado_atual()      matriz exibida no painel de resultado
  self.app.ao_atualizar(cb)       registra cb() para ser chamado quando a exibição mudar
  self.app.status                 StringVar da barra de status
  self.app.notebook               o ttk.Notebook das abas
  self.visivel()                  True se esta aba é a que está selecionada
"""

import tkinter as tk
from tkinter import ttk

ROTULOS_ABAS = {}  # os títulos completos vão no seletor de ferramentas


class Aba:
    titulo = "Aba"

    def __init__(self, app, notebook):
        self.app = app
        self.notebook = notebook
        self.externo = ttk.Frame(notebook)
        notebook.add(self.externo, text=ROTULOS_ABAS.get(self.titulo, self.titulo))

        tela = tk.Canvas(self.externo, highlightthickness=0)
        barra = ttk.Scrollbar(self.externo, orient="vertical", command=tela.yview)
        tela.configure(yscrollcommand=barra.set)
        barra.pack(side="right", fill="y")
        tela.pack(side="left", fill="both", expand=True)

        self.frame = ttk.Frame(tela)
        janela = tela.create_window((0, 0), window=self.frame, anchor="nw")
        self.frame.bind("<Configure>", lambda e: tela.configure(scrollregion=tela.bbox("all")))
        tela.bind("<Configure>", lambda e: tela.itemconfigure(janela, width=e.width))

        def roda(e):
            if self.frame.winfo_reqheight() > tela.winfo_height():
                passo = -1 if (e.num == 4 or getattr(e, "delta", 0) > 0) else 1
                tela.yview_scroll(passo * 2, "units")

        def entrou(_):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                tela.bind_all(ev, roda)

        def saiu(_):
            for ev in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                tela.unbind_all(ev)

        tela.bind("<Enter>", entrou)
        tela.bind("<Leave>", saiu)

        self.construir(self.frame)

    def visivel(self):
        return self.notebook.select() == str(self.externo)

    def construir(self, frame):
        raise NotImplementedError
