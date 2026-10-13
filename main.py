import tkinter as tk
from interface.app import App
from interface.aba_intensidade import AbaIntensidade, AbaLinearPartes
from interface.aba_esteganografia import AbaEsteganografia
from interface.aba_histograma import AbaHistograma
from interface.aba_convolucao import AbaConvolucao

"""Trabalho 1 - Processador de Imagens.

Execução:   python main.py
Instalação: pip install pillow numpy --> Vejo depois se excluo o requirements ou não, vou ver
"""

# Uma aba por ferramenta. Ao implementar uma nova etapa, importe a aba acima
# e acrescente-a aqui.
ABAS = [
    AbaIntensidade,       # Etapa 1
    AbaLinearPartes,      # Etapa 1
    AbaEsteganografia,    # Etapa 1
    AbaHistograma,        # Etapa 2
    AbaConvolucao,        # Etapa 2
]

if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz, ABAS)
    raiz.mainloop()
