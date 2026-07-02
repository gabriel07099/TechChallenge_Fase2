# Wine Quality Classification - Tech Challenge Fase 2

Grupo:
    Armando Caliari Silva - rm372117
   
    Caike Herbe Jauch Soares - rm371123
    
    Gabriel Santos de Oliveira Arruda - rm371178
    
    Gustavo Ferreira da Silva Santana -rm370545
    
    Kauan Lucas Gomes Jardim - rm370438

## Objetivo

Treinar e avaliar modelos de classificação binária capazes de prever se um vinho é "bom" (nota de qualidade ≥ 7) ou "comum" (nota < 7).

## Modelos treinados e resultados

Foram comparados 2 algoritmos:  Naive Bayes e Random Forest:

Melhor modelo: Random Forest com "class_weight='balanced'". O Random Forest supera o Naive Bayes em todas as variantes e métricas.

## Teste prático do modelo

O notebook inclui uma função reutilizável "predict_wine_quality(sample)", que recebe as 11 variáveis físico-químicas de um vinho e retorna a classe prevista com sua probabilidade.

O modelo foi validado em:
- 5 amostras reais do conjunto de teste, obtendo 3 acertos (60%).
- 2 vinhos hipotéticos com perfis opostos, confirmando predições coerentes com as características físico-químicas esperadas.
