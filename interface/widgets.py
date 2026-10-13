import tkinter as tk

"""Componentes de interface reutilizáveis pelas abas."""

def slider(pai, texto, de, ate, ini, res=1, cmd=None):
    """Cria um controle deslizante já posicionado. `cmd` é chamado sem argumentos."""
    s = tk.Scale(pai, from_=de, to=ate, orient="horizontal", label=texto,
                 resolution=res, length=280,
                 command=(lambda v: cmd()) if cmd else None)
    s.set(ini)
    s.pack(fill="x", padx=6)
    return s
