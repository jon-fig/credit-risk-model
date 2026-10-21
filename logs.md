# Log de Alterações

## 2026-10-20 — Configuração do ambiente

- O `.venv` local estava inoperante: o `pyvenv.cfg` apontava para
  `AppData\Local\Programs\Python\Python315\python.exe`, que não existe. O
  arquivo foi corrigido para usar o interpretador instalado em `C:\Python315`.
- Criado `requirements.txt` com as versões instaladas de `pandas`, `numpy`,
  `scikit-learn`, `matplotlib`, `seaborn`, `joblib` e `xgboost`.

## 2026-10-20 — Exploração e limpeza dos dados

- Dataset carregado de `data/cs-training.csv`: 150.000 linhas e 12 colunas
  (incluindo o índice `Unnamed: 0`).
- Removidos a coluna de índice e 609 duplicatas exatas, resultando em 149.391
  registros (6,70% de inadimplentes).
- Ausentes: `MonthlyIncome` (19,56%) e `NumberOfDependents` (2,56%).
- Identificados 225 registros com sentinelas `96/98` em cada coluna de atraso e
  1 registro com `age = 0`, tratados como ausentes e imputados pela mediana.

## 2026-10-20 — Modelagem e avaliação

- `X` (10 variáveis) e `y` (`SeriousDlqin2yrs`) separados; divisão estratificada
  80/20 com `random_state=42` (119.512 treino / 29.879 teste).
- Treinados `RandomForestClassifier` (baseline, com `class_weight`) e
  `XGBClassifier` (avançado, com `scale_pos_weight`), ambos com
  `random_state=42`.
- Random Forest: ROC-AUC 0,8554 | precisão 0,3853 | revocação 0,5025 |
  PR-AUC 0,3809.
- XGBoost: ROC-AUC 0,8614 | precisão 0,2210 | revocação 0,7637 |
  PR-AUC 0,3959 — melhor modelo, salvo em `models/credit_model.pkl`.
- Geradas matrizes de confusão e gráficos de importância das variáveis em
  `reports/figures/`, além de `reports/metrics.json`,
  `reports/model_comparison.csv`, `reports/feature_importance.csv` e
  `reports/data_summary.json`.
- Corrigido problema de serialização: `CreditDataCleaner` foi movido para
  `src/preprocessing.py` para que o pipeline salvo possa ser recarregado em
  outro processo.

## 2026-10-20 — Documentação

- Criado `README.md` com objetivo, dados, estrutura, instruções de execução,
  métricas alcançadas, exemplo de uso do modelo e limitações.
- Pipeline validado de ponta a ponta com `python -m src.pipeline`; o modelo
  salvo foi recarregado e testado com dados brutos contendo ausentes e
  sentinelas.
