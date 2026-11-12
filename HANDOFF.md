# HANDOFF: Tech Challenge Fase 4 (credit-scoring-observability)

Prazo: 17/11/2026. Vídeo e entrega: grupo. Projeto de Gabriel, pensado para portfólio.
Forma de ensino: passo a passo, código pronto e explicado, "porquê" de cada coisa.

## Decisões
- Dataset: UCI Default of Credit Card Clients (30.000 linhas). Giv. Me Some Credit descartado (licença da competição incerta).
- Poetry, Python 3.12 (`>=3.12,<3.14`), pandas fixado em `>=2.2,<3` (Evidently não roda com pandas 3).
- Colunas em minúsculas; `pay_0` renomeada para `pay_1`; alvo = `default` (22,12%).
- `education` 0/5/6 -> 4; `marriage` 0 -> 3 (em `prep.clean`).
- Contrato valida ID único (não linha única: 35 linhas repetidas são clientes distintos).
- Contrato = limites de sanidade, não faixas do treino (drift não pode ser barrado por ele).
- `sex` e `id` fora das features (governança); `sex` fica nos dados para auditoria de viés.
- Referência = 20.000 linhas (`data/reference/reference.csv`); base de produção = 10.000 (`data/production/production_base.csv`).

## Código (src/credit_scoring_observability/)
data.py (carga + perfil), prep.py (limpeza + split), contracts.py (Pandera),
make_bad_batch.py (lote ruim), train.py (baseline XGBoost, aborta se o contrato falhar).
Testes: tests/test_contracts.py.

## Resultado da Etapa 1
Baseline XGBoost: AUC 0,791; recall de inadimplentes 0,382 (corte 0,5). Modelo em `models/baseline.joblib`
(fora do Git, gere com train.py). Métricas em `reports/baseline_metrics.json`.

## Próximo: Etapa 2 (drift)
Gerar produção a partir de `production_base.csv` alterando >= 2 variáveis importantes
(ex.: limit_bal, pay_1/atrasos) + concept drift (mudar a relação feature->alvo);
predizer com o baseline; relatório Evidently (PSI/KS) em `reports/`.
Importar `feature_columns` de train.py. Etapa 3: logs + MLflow ou Prometheus/Grafana. Etapa 4: LGPD, causalidade, README.

## Resultado da Etapa 2
- `simulate_drift.py`: 4 períodos de 2.500 linhas (período 1 = controle). Data drift em bill_amt, pay_1 e age;
  concept drift = bons pagadores (pay_1 <= 0) passam a dar calote. Todos passam no contrato.
- Degradação do baseline: AUC 0,788 -> 0,736 -> 0,694 -> 0,649; recall 0,344 -> 0,267.
- `drift.py`: Evidently com PSI (limite 0,2) e KS (p < 0,05). Saídas: `reports/drift_period_{k}_psi.html`,
  `reports/drift_period_4_ks.html`, `reports/drift_summary.json`, `data/production/predictions.csv`.
- PSI aponta age (período 3) e age + pay_1 (período 4); bill_amt fica em atenção (~0,13). KS é mais sensível.
- Alertas planejados: crítico (PSI >= 0,2 ou queda forte de AUC); atenção (PSI 0,1 a 0,2 ou KS).

## Próximo: Etapa 3 (observabilidade)
Pipeline que processa cada lote (valida contrato -> prediz -> drift), com logs estruturados e métricas centralizadas.