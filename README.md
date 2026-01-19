# UCRPOS: Parkinson's Disease Voice Diagnosis using Hybrid TCN-Transformer Model

## 🎯 Project Overview

UCRPOS is a comprehensive deep learning framework for Parkinson's Disease (PD) detection using voice analysis. This project implements a novel hybrid Temporal Convolutional Network (TCN) and Transformer architecture with advanced causal inference and interpretability features.

## 🔬 Key Features

- **Hybrid Architecture**: Combines TCN and Transformer for sequential audio feature processing
- **Causal Inference**: Implements counterfactual analysis for clinical interpretability
- **Multi-modal Features**: Integrates MFCC sequences and traditional acoustic features
- **Advanced Interpretability**: SHAP analysis and mechanistic interpretability
- **Statistical Validation**: Comprehensive cross-validation and statistical testing
- **Clinical Relevance**: UPDRS correlation analysis for disease severity assessment

## 📋 Requirements

```bash
pip install -r requirements.txt

🚀 Quick Start
1. Data Preparation
Organize your audio data as follows:
复制
data/
├── HC/          # Healthy control audio files (.wav)
└── PD/          # Parkinson's disease audio files (.wav)
2. Basic Usage
Python
复制
from src.data_processing import ParkinsonDataProcessor
from src.models.hybrid_model import build_hybrid_model
from src.training.train_pipeline import TrainPipeline

# Initialize data processor
processor = ParkinsonDataProcessor(
    health_dir='data/HC',
    parkinson_dir='data/PD'
)

# Preprocess data
X_train_seq, X_test_seq, X_train_global, X_test_global, y_train, y_test = processor.preprocess()

# Build and train model
config = {
    "MODEL_ARCH": "HYBRID",
    "USE_CROSS_ATTENTION": True,
    "USE_GLOBAL_MODULATION": True
}

pipeline = TrainPipeline(config)
model, history = pipeline.train(X_train_seq, X_train_global, y_train,
                               X_test_seq, X_test_global, y_test)
3. Advanced Analysis
Python
复制
from src.analysis.causal_analysis import perform_counterfactual_analysis
from src.analysis.shap_analysis import perform_shap_analysis_enhanced

# Causal analysis
causal_effects = perform_counterfactual_analysis(model, X_test_seq, X_test_global, y_test, processor)

# SHAP analysis
shap_results = perform_shap_analysis_enhanced(model, train_data, test_data)
📊 Model Architecture
Hybrid TCN-Transformer Model
TCN Branch: Captures local temporal patterns with dilated convolutions
Transformer Branch: Models long-range dependencies with self-attention
Cross-Attention Fusion: Enables interaction between TCN and Transformer features
Global Feature Modulation: Integrates traditional acoustic features
Feature Engineering
Sequence Features: 40-dimensional MFCC features
Global Features: Traditional acoustic measures (F0, energy, spectral features)
Temporal Processing: Variable-length sequence handling with padding
🔍 Analysis Capabilities
1. Causal Inference
Counterfactual Analysis: Average Causal Effect (ACE) estimation
Bootstrap Confidence Intervals: Statistical significance testing
Mediation Analysis: Feature group effects
Individual Treatment Effects: Patient-level causal attribution
2. Interpretability
SHAP Analysis: Feature importance with statistical testing
Attention Visualization: Temporal attention patterns
Mechanistic Analysis: Individual sample interpretation
3. Statistical Validation
K-Fold Cross-Validation: Robust performance estimation
Non-parametric Testing: Wilcoxon signed-rank tests
Effect Size Analysis: Cohen's d and correlation measures
Multiple Comparison Correction: Bonferroni adjustment
📈 Performance
Dataset Statistics
Total Samples: Varies by dataset
Features: 40 MFCC + 10 traditional acoustic features
Classes: Healthy Control vs. Parkinson's Disease
Sampling Rate: 22.05 kHz
Model Comparison
The proposed hybrid model demonstrates superior performance compared to:
Traditional ML: SVM, Random Forest, Logistic Regression
Deep Learning: CNN-LSTM, Pure Transformer, TCN-only
State-of-the-art methods from recent literature
🛠️ Project Structure
复制
UCRPOS/
├── src/                    # Source code
│   ├── data_processing.py  # Data loading and preprocessing
│   ├── models/             # Model architectures
│   ├── analysis/           # Analysis modules
│   ├── visualization/      # Plotting functions
│   ├── training/           # Training pipelines
│   └── utils/              # Utility functions
├── config/                 # Configuration files
├── data/                   # Dataset directory
├── results/                # Output results
├── main.py                 # Main execution script
└── examples/               # Usage examples
📊 Visualization Outputs
The framework generates comprehensive visualizations:
Training Curves: Loss and AUC progression
Causal Analysis: ACE effects with confidence intervals
SHAP Analysis: Feature importance and distributions
Model Comparison: Statistical significance matrices
Attention Patterns: Temporal attention visualization
🔬 Research Applications
Clinical Research
Diagnostic Support: Automated PD screening
Disease Monitoring: Progression tracking
Feature Discovery: Biomarker identification
Treatment Evaluation: Therapy response assessment
Academic Research
Method Development: Architecture optimization
Feature Engineering: Novel acoustic measures
Causal Inference: Clinical decision support
Interpretability: Explainable AI for healthcare
📚 Citation
If you use this code in your research, please cite:
bibtex
复制
@article{ucrpos2024,
  title={Hybrid TCN-Transformer Model for Parkinson's Disease Voice Diagnosis with Causal Inference},
  author={Your Name},
  journal={Journal Name},
  year={2024}
}
🤝 Contributing
Fork the repository
Create a feature branch (git checkout -b feature/amazing-feature)
Commit your changes (git commit -m 'Add some amazing feature')
Push to the branch (git push origin feature/amazing-feature)
Open a Pull Request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
📞 Contact
For questions and collaboration:
Email: your.email@example.com
Institution: Guangdong Pharmaceutical University
🙏 Acknowledgments
Guangdong Medical Research Foundation
Biomedical Engineering Journal reviewers
Parkinson's disease research community
📖 References
Key references are integrated throughout the code with citations:
[1] TCN architecture papers
[2] Transformer applications in healthcare
[3] Traditional ML methods for PD detection
[4] Causal inference in medical AI
[5] SHAP analysis for deep learning
