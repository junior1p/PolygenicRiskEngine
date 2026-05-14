# PolygenicRiskEngine

Pure Python polygenic risk score analysis pipeline.

## Features
- GWAS summary statistics simulation
- LD clumping (r2<0.1, 500kb window)
- C+T PRS computation
- P-value threshold optimization (AUC)
- Population stratification PCA
- Risk percentile odds ratios

## Usage
```bash
pip install numpy scipy pandas matplotlib
python polygenic_risk_engine.py
```

## Results (1000 individuals, 5000 SNPs, 3 diseases)
- T2D AUC=0.55 at p<5e-8 threshold
- Top 10% vs bottom 10% OR=1.49
- PC1 explains 0.3% variance
