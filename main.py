import tkinter as tk
from interface.app import App
from interface.aba_intensidade import AbaIntensidade, AbaLinearPartes
from interface.aba_esteganografia import AbaEsteganografia
from interface.aba_histograma import AbaHistograma
from interface.aba_convolucao import AbaConvolucao
from interface.aba_bordas import AbaBordas
from interface.aba_nitidez import AbaNitidez
from interface.aba_geometria import AbaGeometria
from interface.aba_frequencia import AbaFrequencia
from interface.aba_cor import AbaCor

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
    AbaNitidez,           # Etapa 2/3 (Laplaciano, unsharp, high-boost)
    AbaBordas,            # Etapa 3
    AbaGeometria,         # Etapa 3
    AbaFrequencia,        # Etapas 4 e 5
    AbaCor,               # Etapa de cor
]

if __name__ == "__main__":
    raiz = tk.Tk()
    App(raiz, ABAS)
    raiz.mainloop()
