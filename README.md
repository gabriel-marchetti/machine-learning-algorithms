# Temis - Machine Learning Algorithms

Um projeto Python para algoritmos de machine learning com métricas de fairness.

## Dependências

As principais bibliotecas utilizadas são:
- **numpy**: Computação numérica
- **pandas**: Manipulação e análise de dados
- **scikit-learn**: Ferramentas de machine learning
- **jax**: Computação numérica com diferenciação automática
- **matplotlib**: Visualizações

## Instalação

### Opção 1: Instalação via setup.py

```bash
pip install -e .
```

Este comando instalará o pacote `Temis` e todas as suas dependências no modo desenvolvimento.

### Opção 2: Instalação via requirements.txt

```bash
pip install -r requirements.txt
```

Este comando instalará apenas as dependências sem instalar o pacote.

### Opção 3: Instalação manual

```bash
pip install numpy pandas scikit-learn jax jaxlib matplotlib
```

## Estrutura do Projeto

- `Temis/` - Pacote principal contendo:
  - `LogisticRegression.py` - Implementação de regressão logística com JAX
  - `dataloader.py` - Classes para carregamento de dados
  - `preprocess_data.py` - Funções de pré-processamento
  - `metrics/` - Métricas de desempenho (accuracy, precision, recall, f1, brier_score)
  - `fairness_metrics/` - Métricas de fairness (spd, aod, aaod, dir)
  - `math_utils/` - Funções matemáticas auxiliares
  - `comparison_utils/` - Utilitários para comparação

- `notebooks/` - Jupyter notebooks para experimentos
- `datasets/` - Datasets utilizados nos experimentos
- `tests/` - Testes unitários

## Desenvolvimento

Para contribuir ao projeto:

1. Instale o pacote em modo desenvolvimento:
   ```bash
   pip install -e .
   ```

2. Execute os testes:
   ```bash
   python -m pytest tests/
   ```

3. Ou execute testes específicos:
   ```bash
   python tests/LogisticRegressionTest.py
   ```
