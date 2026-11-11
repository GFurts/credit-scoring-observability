# Credit Scoring Observability

Camada de sustentação para um modelo de credit scoring: contratos de dados,
simulação e detecção de drift, observabilidade e governança (LGPD).
Projeto do Tech Challenge Fase 4 (FIAP MLET).

## Dados
Dataset *Default of Credit Card Clients* (UCI Machine Learning Repository),
de Yeh & Lien (2009). O arquivo bruto **não** é versionado neste repositório.
Baixe-o no site da UCI e salve como `data/raw/default of credit card clients.xls`.

## Como rodar
    poetry install
    poetry run python -m credit_scoring_observability.data
    poetry run python -m credit_scoring_observability.prep
    poetry run python -m credit_scoring_observability.contracts
    poetry run python -m credit_scoring_observability.train

## Status
- [x] Etapa 1: contrato de dados e baseline
- [ ] Etapa 2: simulação e detecção de drift
- [ ] Etapa 3: observabilidade
- [ ] Etapa 4: governança e LGPD