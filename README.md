# AI Cliff: Scaling Laws and Innovation Economics in Frontier AI

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

> **Research on AI scaling laws, knowledge diffusion, and investment economics in frontier AI systems**

This repository contains the code, data, and analysis for studying the economics of frontier AI model development, focusing on scaling laws, knowledge diffusion between proprietary and open models, and investment dynamics.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Research Questions](#key-research-questions)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Data Sources](#data-sources)
- [Outputs](#outputs)
- [Replication](#replication)
- [Citation](#citation)
- [License](#license)

## 🔍 Overview

This project investigates three fundamental aspects of frontier AI development:

1. **Scaling Laws**: Empirical relationships between model parameters, training compute, and costs
2. **Knowledge Diffusion**: Transfer patterns between proprietary and open-source AI models
3. **Investment Economics**: Net Present Value (NPV) analysis and viability of frontier AI projects across different tiers

The analysis combines econometric methods (OLS, survival analysis) with Monte Carlo simulations to provide insights into the economic sustainability and diffusion dynamics of large-scale AI investments.

## 🎯 Key Research Questions

- How do AI model parameters scale with training compute and costs?
- What are the temporal patterns of knowledge diffusion from proprietary to open models?
- What is the expected NPV and risk profile of frontier AI investments?
- How do different organizational types (academia, industry, government) differ in their innovation patterns?

## 📂 Project Structure

```
AI_Cliff/
├── data/                           # Data files
│   ├── scaling_dataset.parquet     # Model scaling data
│   └── diffusion_pairs.parquet     # Diffusion analysis pairs
├── Epoch.AI/                       # Raw Epoch AI data
│   ├── large_scale_ai_models.csv
│   ├── notable_ai_models.csv
│   └── ml_hardware.csv
├── AI_Diffusion_Economics_v2.py    # Main analysis script
├── Difussion-1.ipynb               # Interactive analysis notebook
├── Data_AI-7.ipynb                 # Data preparation notebook
├── config_ai_diffusion.yaml        # Configuration parameters
├── requirements_ai_notebook.txt    # Python dependencies
├── artifacts/                      # Generated outputs
│   ├── figures/                    # Analysis visualizations
│   ├── tables/                     # Statistical tables
│   ├── logs/                       # Execution logs
│   └── models/                     # Saved model objects
├── academic_outputs/               # Publication-ready outputs
│   ├── figures/                    # Paper figures
│   ├── tables/                     # Paper tables
│   ├── replication/                # Replication package
│   └── validation/                 # Robustness checks
└── create_paper_figures*.py        # Figure generation scripts
```

## 🔧 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/AI_Cliff.git
cd AI_Cliff
```

2. **Install dependencies**:
```bash
pip install -r requirements_ai_notebook.txt
```

Required packages:
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `statsmodels` - Statistical modeling
- `lifelines` - Survival analysis
- `matplotlib` - Visualization
- `pyyaml` - Configuration management (optional)

## 🚀 Usage

### Quick Start

1. **Run the main analysis pipeline**:
```bash
python AI_Diffusion_Economics_v2.py
```

This will:
- Load data from `./data/` directory
- Perform scaling laws regression
- Conduct survival analysis (Kaplan-Meier and Cox PH)
- Run Monte Carlo NPV simulations
- Generate all figures and tables in `./artifacts/`

2. **Interactive analysis via Jupyter**:
```bash
jupyter notebook Difussion-1.ipynb
```

### Configuration

All parameters are controlled via `config_ai_diffusion.yaml`:

```yaml
paths:
  scaling_data: "./data/scaling_dataset.parquet"
  diffusion_data: "./data/diffusion_pairs.parquet"

scaling_spec:
  compute_unit: auto
  params_scale: 1e9

diffusion_spec:
  window_months: 60
  min_lead_months: 4.8

investment_spec:
  price_per_flop_now: 5e-11
  annual_discount_rate: 0.20
  horizon_months: 72
  n_sims: 10000
```

Modify these parameters to adjust the analysis specifications.

## 📊 Methodology

### 1. Scaling Laws Analysis

**Approach**: Log-log OLS regression with controls
- **Dependent variables**: Model parameters, training compute, costs
- **Controls**: Year fixed effects, architecture type, organization tier
- **Estimation**: Robust standard errors, bootstrap confidence intervals

**Key specifications**:
```
log(Parameters) ~ log(Compute) + Year FE + Architecture FE
log(Cost) ~ log(Parameters) + Year FE + Architecture FE
```

### 2. Diffusion Analysis

**Approach**: Survival analysis (Kaplan-Meier and Cox Proportional Hazards)
- **Event**: Open model release following proprietary model
- **Time**: Months between proprietary and open releases
- **Covariates**: Model tier, organization type, parameter size

**Methods**:
- Kaplan-Meier survival curves by tier
- Cox PH regression with time-varying covariates
- Competing risks analysis

### 3. Investment Economics

**Approach**: Monte Carlo simulation of NPV
- **Revenue modeling**: Log-normal distributions calibrated to tier
- **Cost structure**: Training compute × price per FLOP
- **Competition effects**: Post-diffusion revenue drop (30-70%)
- **Risk metrics**: Mean NPV, VaR (5%), probability of positive NPV

**Scenarios analyzed**:
- Tier III: 1T parameters, baseline economics
- Tier IV: 1T parameters, higher costs (1.8× multiplier)

## 📈 Data Sources

This analysis uses publicly available data from:

**[Epoch AI](https://epochai.org/data)** (CC BY 4.0):
- Large-scale AI models database
- Training compute estimates
- Hardware efficiency benchmarks
- Organization classifications

**Data fields**:
- Model parameters (billions)
- Training compute (FLOPs)
- Training costs (USD)
- Release dates
- Developer organization
- Model architecture
- License type (proprietary/open)

## 📤 Outputs

### Figures

**Main Analysis**:
- `Figure_01_Scaling_Laws.png` - Parameter vs compute scaling relationships
- `Figure_02_Diffusion_Analysis.png` - Knowledge diffusion survival curves
- `Figure_3_Compute_Scaling_by_Tier.png` - Compute scaling by organization tier
- `Figure_4_Survival_Function_AI_Innovation.png` - KM survival curves
- `Figure_5_NPV_Distributions_by_Tier.png` - NPV distributions
- `Figure_6_Welfare_Analysis_Policy_Intervention.png` - Welfare analysis

### Tables

**Econometric Results**:
- `Table_01_Summary_Statistics.txt` - Descriptive statistics
- `Table_02_Parameter_Estimates.txt` - Regression coefficients
- `scaling_table_parameters_vs_training_compute.txt` - Scaling law estimates
- `scaling_table_parameters_vs_training_cost.txt` - Cost relationship estimates

### Validation Reports

- `data_loading_report.csv` - Data quality checks
- `imputation_report.json` - Missing data imputation diagnostics
- `robustness_analysis_report.json` - Sensitivity analysis results

## 🔄 Replication

The `academic_outputs/replication/` directory contains a complete replication package:

1. **Datasets**: Pre-processed and imputed data files
2. **Code**: Self-contained analysis scripts
3. **Results**: Summary statistics and estimates
4. **Documentation**: `README.md` with step-by-step instructions

To replicate the main findings:

```bash
cd academic_outputs/replication/
jupyter notebook replication_notebook.ipynb
```

**System requirements**:
- Python 3.8+
- 8GB+ RAM recommended
- Processing time: ~30-60 minutes

## 📝 Citation

If you use this code or data in your research, please cite:

```bibtex
@article{,
  title={Scaling Laws and Knowledge Diffusion in Frontier AI Innovation},
  author={},
  year={2025},
  journal={[Journal Name]},
  note={Working Paper}
}
```

See `academic_outputs/replication/CITATION.txt` for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Data Usage**: The Epoch AI data is used under CC BY 4.0 license. Please cite the original sources when using derived datasets.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/yourusername/AI_Cliff/issues).

## 👥 Authors

**XXX XXX* - *YYYY*

## 🙏 Acknowledgments

- [Epoch AI](https://epochai.org/) for providing comprehensive AI model data
- Contributors to the open-source packages used in this analysis
- Research institutions supporting this work



---

**Last Updated**: October 2025  
**Version**: 2.0  
**Status**: Active Research Project

