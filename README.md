# Paralelização e Avaliação de Desempenho do Algoritmo de Colônia de Formigas
Este trabalho tem como objetivo paralelizar o algoritmo de colônia de formigas(ACO), uma técnica de otimização inspirada no comportamento das formigas reais, e avaliar a qualidade dos resultados obtidos após a paralelização.

A versão sequencial do algoritmo ACO foi inicialmente implementada por Henrique R. Hott em seu TCC, utilizando uma única thread para executar as tarefas. Posteriormente, foi explorada uma abordagem alternativa de paralelização utilizando a biblioteca joblib e a função Parallel, que utiliza paralelismo de thread, com o objetivo de acelerar o tempo de execução do algoritmo.

Após a paralelização, foram conduzidos experimentos comparativos entre a versão sequencial e a versão paralelizada do algoritmo. Esses experimentos consideraram diferentes aspectos, como o tempo de execução, a utilização de recursos computacionais e, principalmente, a qualidade dos resultados obtidos.

A avaliação da qualidade dos resultados foi realizada utilizando algoritmos de Machine Learning(ML), incluindo RandomForest, SVM e Árvore de Decisão. Esses algoritmos foram aplicados aos conjuntos de dados gerados pelo algoritmo de colônia de formigas, tanto na versão sequencial quanto na versão paralelizada.

# Estrutura

Este projeto adota a seguinte estrutura:
- colonia
  - ACO
  - Parallel ACO
- machine_learning
  - RandomForest
  - SVM
  - Tree
  - Decision Tree
- graphics
  - Line Graphics
  - Bar Graphs
- articles
  - tcc

## Informações 

Este trabalho é o resultado de uma pesquisa realizada como parte do meu Trabalho de Conclusão de Curso (TCC) e da minha participação em um projeto de iniciação científica na área de Machine Learning, com foco na seleção de instâncias.

O objetivo principal do estudo foi investigar e analisar a técnica de seleção de instâncias ACO em conjuntos de dados, visando melhorar o desempenho e a eficiência de algoritmos de Machine Learning.

Para obter mais detalhes sobre o estudo realizado, o artigo correspondente pode ser encontrado no diretório "articles", fornecendo informações adicionais e aprofundadas sobre as metodologias, os experimentos conduzidos e os resultados obtidos.
