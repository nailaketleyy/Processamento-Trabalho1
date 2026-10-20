import tkinter as tk
from tkinter import ttk

"""Componentes de interface reutilizáveis pelas abas."""


def _margem_minima(valor, minimo):
    partes = tuple(max(int(parte), minimo) for parte in str(valor).split())
    return partes[0] if len(partes) == 1 else partes


def espacamento_botao(evento):
    widget = evento.widget
    if widget.winfo_manager() == "pack":
        info = widget.pack_info()
        widget.pack_configure(
            padx=_margem_minima(info["padx"], 4),
            pady=_margem_minima(info["pady"], 4),
        )

def slider(pai, texto, de, ate, ini, res=1, cmd=None):
    """Cria um controle deslizante já posicionado. `cmd` é chamado sem argumentos."""
    s = tk.Scale(pai, from_=de, to=ate, orient="horizontal", label=texto,
                 resolution=res, length=280,
                 command=(lambda v: cmd()) if cmd else None)
    s.set(ini)
    s.pack(fill="x", padx=6)
    return s
