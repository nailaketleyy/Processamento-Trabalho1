# Processador de Imagens — Trabalho 1

Aplicação desktop em Python para explorar operações de processamento digital
de imagens relacionadas ao capítulo 3 de *Processamento Digital de Imagens*,
3ª edição, de Rafael C. Gonzalez e Richard E. Woods. A interface usa Tkinter;
Pillow é usado para abrir, exibir e salvar imagens, e NumPy para os cálculos.

## Executar

```powershell
python -m pip install -r requirements.txt
python main.py
```

Abra uma imagem pelo menu **Arquivo > Abrir**. O painel à direita mostra a
imagem original e o resultado/pré-visualização. Os botões **Aplicar** gravam a
operação no histórico; **Desfazer** e **Restaurar original** permitem voltar.
Ao salvar, escolha BMP ou PNG quando precisar de um formato sem perdas.

## Operações de intensidade

Na aba **Intensidade**:

- **Negativo:** tabela de consulta `s = 255 - r`.
- **Brilho:** soma um deslocamento `b` aos níveis de intensidade, saturando no
  intervalo `[0, 255]`.
- **Potência (gama):** aplica `s = c r^γ`, com a entrada normalizada em
  `[0, 1]`; `c` e `γ` são ajustáveis.
- **Limiarização:** converte a luminância em uma imagem binária usando `T`.

Na aba **Linear por partes**, os pontos de controle definem a transformação
linear por trechos e o gráfico mostra a curva. Também há alargamento automático
de contraste, fatiamento de níveis e extração de planos de bits.

## Esteganografia

A aba **Esteganografia** oculta texto ou bytes de um arquivo nos bits menos
significativos dos canais RGB e também permite extrair os dados. O cabeçalho
armazena a assinatura, a quantidade de bits usados e o tamanho dos dados.
Aumentar os bits por canal aumenta a capacidade, mas também a alteração
perceptível. JPEG é com perdas e pode destruir a mensagem; use BMP ou PNG.

## Histograma e equalização

A aba **Histograma** exibe a distribuição das 256 intensidades para luminância,
intensidade HSI, para cada canal RGB ou os três canais simultaneamente. A escala logarítmica
ajuda a visualizar valores muito desiguais; média e desvio padrão da
luminância também são exibidos. A equalização usa a soma acumulada do
histograma para construir a transformação. Pode ser aplicada à luminância
(preservando melhor as cores), a cada canal RGB ou após converter para cinza.

## Filtragem espacial

Na aba **Convolução**, escolha uma vizinhança quadrada ímpar entre `3×3` e
`21×21`, selecione um filtro ou edite seus coeficientes, e pré-visualize antes
de aplicar. Convolução gira o kernel em 180°; correlação aplica o kernel sem
girá-lo. Ambas usam padding configurável (**zeros**, **replicar borda** ou
**espelhar**) e preservam largura e altura da imagem.

Filtros disponíveis:

- **Média (caixa):** média aritmética local; selecione a divisão pela soma dos
  coeficientes para manter o nível médio.
- **Gaussiano:** média ponderada com pesos Gaussianos; `n` e `σ` são ajustáveis.
- **Média ponderada (1-2-1):** kernel clássico `[[1,2,1],[2,4,2],[1,2,1]]`.
- **Mediana:** filtro não linear que substitui cada pixel pela mediana da
  vizinhança escolhida, útil para ruído impulsivo.
- **Aguçamento Laplaciano:** aplica `f - ∇²f`, usando o Laplaciano de quatro
  vizinhos.
- **High-boost:** aplica `A f - f_gauss`; `A ≥ 1` controla a quantidade de
  detalhes realçados e a vizinhança/`σ` controlam a suavização usada na máscara.
- Também estão disponíveis Laplaciano de bordas, Sobel, identidade e um kernel
  personalizado.

As operações de convolução são calculadas em ponto flutuante. O seletor
**Saída** converte o resultado para 8 bits: **saturar** limita os valores a
`[0,255]`, **valor absoluto** é útil para respostas derivativas e
**normalizar apenas na prévia** mapeia o mínimo/máximo para `[0,255]` sem
gravar a normalização ao aplicar o filtro. A mediana é não linear, ordena os
valores da janela e não usa coeficientes de kernel.

## Sobel e gradiente

A aba **Sobel / Gradiente** aplica o operador de Sobel à luminância, separando
as respostas direcionais `Gx` (Sobel X) e `Gy` (Sobel Y). As duas respostas são
normalizadas de forma independente para `[0,255]` para visualização em tons de
cinza. A opção **Magnitude do gradiente** combina as componentes por
`sqrt(Gx² + Gy²)` e normaliza a magnitude; essa combinação é não linear. A
regra de padding das bordas também pode ser selecionada.

## Transformações geométricas

A aba **Geometria** aplica escala horizontal/vertical, rotação e translação por
mapeamento inverso: para cada pixel do destino, calcula-se a coordenada de
amostragem na imagem de origem. A rotação positiva é no sentido horário da
tela. O pivô padrão é o centro da imagem, mas pode ser definido em coordenadas
`(x,y)`; a translação usa `dx` para direita/esquerda e `dy` para baixo/cima.

A saída preserva largura e altura da imagem por padrão. **Expandir saída**
calcula um quadro que contém a transformação; o processamento é recusado se a
saída ultrapassar 16 milhões de pixels. Regiões que caem fora da imagem de
origem são preenchidas de preto. **Vizinho mais próximo** escolhe o pixel de
origem mais próximo; **bilinear** calcula uma média ponderada dos quatro
vizinhos. Use **Pré-visualizar** antes de aplicar a transformação.

## Transformada de Fourier e aliasing

A aba **Frequência / Aliasing** mostra o espectro da luminância e aplica a
mesma máscara de frequência separadamente aos canais RGB. A DFT/IDFT própria
tem limite de 4096 pixels; a FFT radix-2 própria aceita dimensões que sejam
potências de 2, e **FFT NumPy** aceita dimensões arbitrárias. O espectro aparece
deslocado ao centro e em escala logarítmica.

Estão disponíveis os filtros ideais **passa-baixa**, **rejeita-baixa**,
**passa-faixa**, **rejeita-faixa**, **passa-notch** e **rejeita-notch**. Seus
cortes são distâncias em pixels no plano de frequências. Também se pode editar
o espectro com um pincel: nível 0 rejeita, nível 1 mantém, e valores
intermediários atenuam; o modo suave produz uma transição gradual. Os filtros
notch aceitam vários centros, cada um aplicado junto ao seu simétrico. A IDFT
reconstrói a imagem e oferece saturação ou normalização para exibição.

O botão de convolução circular demonstra
`DFT(f circular-convolução h) = DFT(f) · DFT(h)` com kernel média 3×3. A
demonstração de aliasing subamostra e amplia a imagem aberta; o passa-baixa
opcional antes da subamostragem reduz aliasing. O botão **Gerar padrão de
Moiré** cria duas grades sinusoidais de frequências próximas, cuja superposição
forma franjas.

## Operações de cor

A aba **Cor RGB / HSI** converte RGB↔HSI, equaliza o componente de intensidade
HSI e ajusta matiz, saturação e brilho. Também oferece escala de cinza por
média simples ou luminância ponderada, negativo, ajustes dos pares C/R, M/G e
Y/B, sépia e chroma-key. No chroma-key, escolha a imagem substituta e ajuste a
distância RGB máxima em relação ao verde puro; o fundo é redimensionado por
vizinho mais próximo.

## Organização do código

- `algoritmos/`: operações independentes da interface.
- `interface/`: aplicação Tkinter e abas de ferramentas.
- `main.py`: registro das abas e ponto de entrada.

O módulo `algoritmos/convolucao.py` implementa convolução, correlação,
padding, criação de kernels, mediana e conversão de resultados para exibição.
`algoritmos/bordas.py`, `algoritmos/geometria.py`, `algoritmos/frequencia.py`
e `algoritmos/cor.py` implementam Sobel/gradiente, geometria, Fourier/aliasing
e operações de cor, respectivamente.
