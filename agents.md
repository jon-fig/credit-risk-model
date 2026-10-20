# Instruções para o Agente & Especificação do Projeto: Modelo de ML para Risco de Inadimplência

## Papel & Objetivo
Você é um Cientista de Dados e Engenheiro MLOps especialista. Sua tarefa é construir um pipeline robusto de classificação binária para prever o risco de inadimplência de crédito usando principalmente Python, Pandas e Scikit-Learn.

## Regras Estritas & Restrições
1. Ambiente: Sempre assuma que um ambiente virtual Python (venv) local está ativo. Não instale pacotes globalmente.
2. Gerenciamento de Arquivos: Salve os modelos gerados no diretório `models/` usando `joblib`.
3. Reproduzibilidade: Sempre defina `random_state=42` nas divisões do `train_test_split` e na inicialização dos modelos.
4. Registro de Logs: Após a conclusão de cada tarefa principal, adicione um breve resumo das alterações ao arquivo `logs.md`.

## Plano de Execução
1. Configuração: Gere um arquivo `requirements.txt` contendo `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib` e `xgboost`.
2. Exploração & Limpeza de Dados: Carregue o dataset em `data/` com Pandas, verifique valores ausentes, realize imputações (mediana/média) e remova colunas desnecessárias.
3. Engenharia de Recursos (Feature Engineering): Separe as variáveis explicativas (X) da variável alvo (y). Execute a divisão dos dados com `train_test_split`.
4. Treinamento do Modelo: Treine dois modelos para comparar um baseline vs. avançado. RandomForestClassifier (Scikit-Learn) e XGBClassifier (XGBoost).
5. Avaliação & Visualização do Modelo:
- Use Matplotlib e Seaborn para plotar a importância das variáveis (feature importance) e matrizes de confusão.
- Calcule as métricas de classificação: pontuação ROC-AUC, Precisão (Precision) e Revocação (Recall).
6. Salvamento de Artefatos: Use `joblib` para salvar o modelo treinado com a melhor performance em `models/credit_model.pkl`.
7. Documentação: Gere um rascunho inicial do arquivo `README.md` descrevendo o objetivo do projeto, métricas alcançadas e como executar o código.