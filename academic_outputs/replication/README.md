# Frontier AI Innovation Economics - Replication Package

This package contains all materials needed to replicate the empirical analysis presented in:

**"Scaling Laws and Knowledge Diffusion in Frontier AI Innovation"**

## Contents

- `analysis_dataset.csv` - Main analysis dataset
- `analysis_dataset_imputed.csv` - Dataset after intelligent imputation
- `replication_notebook.ipynb` - Complete analysis code
- `results_summary.json` - Summary of all empirical results
- `README.md` - This file
- `CITATION.txt` - Citation information

## Data Sources

The analysis uses data from:
- Epoch AI Database (https://epochai.org/data)
- Model parameter counts and training costs
- Organization classifications

## Methodology

### Scaling Laws Analysis
- Log-log regressions with robust standard errors
- Bootstrap confidence intervals (10,000 iterations)
- Multiple model specifications for robustness

### Diffusion Analysis
- Multi-dimensional model similarity scoring
- Temporal lag analysis between proprietary and open models
- Causal identification through multiple strategies

### Missing Data Treatment
- Domain-specific imputation using scaling laws
- MICE (Multiple Imputation by Chained Equations)
- Sensitivity analysis and validation

## Replication Instructions

1. Install required packages:
```python
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn
```

2. Run the notebook:
```
jupyter notebook replication_notebook.ipynb
```

3. Results will be generated in respective output directories

## System Requirements

- Python 3.8+
- Jupyter Notebook
- 8GB+ RAM recommended for full analysis
- Processing time: ~30-60 minutes

## Contact

For questions about replication, contact: [Your Contact Info]

Generated: 2025-09-18 18:41:50

## License

This replication package is provided under [Your License] license.
Data usage subject to original sources' terms and conditions.