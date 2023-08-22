import matplotlib.pyplot as plt
import numpy as np

# RANDOM -------------------------------------------------------------
# Valores de acurácia para cada experimento da bae original
# acuracia_base_original = [0.55, 1, 0.87, 0.95, 0.44, 0.58, 0.77, 0.94, 0.72, 0.86, 0.51, 0.61]

# # Valores de acurácia para cada experimento da base reduzida
# acuracia_base_reduzida = [0.80, 0.89, 0.75, 0.91, 0.49, 0.67, 0.71, 0.90, 0.69, 0.83, 0.50, 0.59]

# # Valores de acurácia para cada experimento da base reduzida com a paralelas
# acuracia_base_reduzida_paralela = [0.47, 1, 0.81, 0.89, 0.42, 0.61, 0.75, 0.89, 0.68, 0.85, 0.53, 0.51]

# # Nomes dos experimentos
# nomes_experimentos = ['Post Operative', 'Breast Cancer Coimbra', 'Heart Failure', 'Ionosphere', 'Cirrhosis',
#                       'Data Science', 'Diabetes', 'TicTacToe', 'Cyber Security', 'Titanic', 'Contraceptive', 'Yeast']


# TREE -------------------------------------------------------------
# Valores de acurácia para cada experimento da bae original
acuracia_base_original = [0.59, 0.97, 0.80, 0.90, 0.36, 
                          0.69, 0.70, 0.71, 0.49, 0.81, 0.54, 0.47]

# Valores de acurácia para cada experimento da base reduzida
acuracia_base_reduzida = [0.60, 0.89, 0.68, 0.91, 0.47, 
                          0.59, 0.64, 0.70, 0.46, 0.83, 0.44, 0.40]

# Valores de acurácia para cada experimento da base reduzida com a paralelas
acuracia_base_reduzida_paralela = [0.47, 1, 0.81, 0.82, 0.20, 
                                   0.56, 0.73, 0.77, 0.49, 0.78, 0.46, 0.49]

# Nomes dos experimentos
nomes_experimentos = ['Post Operative', 'Breast Cancer Coimbra', 'Heart Failure', 'Ionosphere', 'Cirrhosis',
                      'Data Science', 'Diabetes', 'TicTacToe', 'Cyber Security', 'Titanic', 'Contraceptive', 'Yeast']

# Configurações do gráfico de linha
# plt.figure(figsize=(7, 5))
# plt.plot(nomes_experimentos, acuracia_base_original, marker='o', label='Base Original')
# plt.plot(nomes_experimentos, acuracia_base_reduzida, marker='o', label='Base reduzida com a colonia')
# plt.plot(nomes_experimentos, acuracia_base_reduzida_paralela, marker='o', label='Base reduzida com a colonia em paralela')
# plt.ylim(0.0, 1.0)  # Define os limites do eixo y
# plt.xlabel('Bases')
# plt.ylabel('Acurácia')
# plt.title('Comparação de Acurácia no algoritimo Random Forest entre as Bases de Dados')
# plt.legend()

# Diminui o tamanho dos labels do eixo x
# plt.xticks(fontsize=8)



# Configurações do gráfico de barra
bar_width = 0.20
index = np.arange(len(nomes_experimentos))

plt.figure(figsize=(8, 6))
plt.bar(index, acuracia_base_original, bar_width, label='Base Original')
plt.bar(index + bar_width, acuracia_base_reduzida, bar_width, label='Base reduzida com a colonia')
plt.bar(index + 2*bar_width, acuracia_base_reduzida_paralela, bar_width, label='Base reduzida com a colonia em paralelizada')
plt.ylim(0.0, 1.0)
plt.xlabel('Bases')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia no algoritimo Árvore de Decisão entre as Bases de Dados')
plt.legend()

# Diminui o tamanho dos labels do eixo x
plt.xticks(index + bar_width/2, nomes_experimentos, fontsize=8)


# Exibe o gráfico
plt.show()

