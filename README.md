# AdaptivePrivFL: Enhanced Federated Deep Learning for Breast Cancer Diagnosis

This repository contains the implementation of "AdaptivePrivFL," an enhanced federated deep learning framework for privacy-preserving breast cancer diagnosis, incorporating adaptive client management and robust differential privacy.

## Project Overview

This project demonstrates an advanced federated learning system to train deep learning models for breast cancer diagnosis while preserving patient privacy and addressing client heterogeneity. The implementation includes:

- An FL framework ("AdaptivePrivFL") with **novel strategic client selection** (based on data quality, historical contribution, and exploration-exploitation).
- **Enhanced adaptive federated averaging** using multi-factor weighting (data size, quality, contribution).
- **Rigorous Differential Privacy** using the **Opacus library** for privacy guarantees during client training.
- **Knowledge Distillation (KD)** for advanced model personalization on client data.
- A conceptual exploration of **layer-specific privacy sensitivity** within the client model.
- Performance evaluation including global model accuracy, F1-score, AUC-ROC, personalization uplift, client contribution analysis, privacy budget tracking, and comparison with centralized approaches.
- Analysis of the privacy-utility tradeoff and model robustness.


## Dataset

This project uses the Wisconsin Breast Cancer Dataset, which is included in scikit-learn and can be loaded using `sklearn.datasets.load_breast_cancer()`. The dataset contains features computed from digitized images of breast mass fine needle aspirates (FNA) and classifies samples as malignant or benign.

## Requirements

- Python 3.8+
- PyTorch 1.9.0+
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- **opacus**

## Setup Instructions

### Option 1: Using a Virtual Environment (Recommended)

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/federated-breast-cancer-diagnosis.git
   cd federated-breast-cancer-diagnosis
   ```

2. **Create and activate a virtual environment**:

   **For Windows**:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

   **For macOS/Linux**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Register the virtual environment with Jupyter**:
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name=federated_dl_env --display-name="Federated DL Environment"
   ```

5. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

6. **Open the notebook and select the kernel**:
   - Open `federated_learning.ipynb`
   - From the Kernel menu, select "Change kernel" and choose "Federated DL Environment"

### Option 2: Using an Existing Python Environment

If you already have Python with the required packages installed:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Nerd-coderZero/federated-breast-cancer-diagnosis.git
   cd federated-breast-cancer-diagnosis
   ```

2. **Install any missing required packages**:
   ```bash
   pip install numpy pandas matplotlib scikit-learn torch opacus
   ```

3. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook
   ```

4. **Open the notebook**:
   - Open `federated_learning.ipynb`

### Option 3: Using Google Colab

If you prefer to run the notebook in the cloud:

1. Download the `federated_learning.ipynb` file from this repository
2. Go to [Google Colab](https://colab.research.google.com/)
3. Click on "Upload" and select the downloaded notebook
4. Run the cells one by one (the first cell should install all required dependencies)

## Running the Notebook

1. **Open the notebook** in Jupyter or your preferred environment
2. **Run the cells sequentially** by pressing Shift+Enter or using the "Run" button
3. The script is organized into sections:
   - Data loading and preprocessing.
   - Definitions of novel FL components (Data Quality, Client Selection, Adaptive Aggregation, KD).
   - Federated learning process with Opacus DP.
   - Model training, personalization, and evaluation.
   - Results visualization and saving.

## Implementation Details

The "AdaptivePrivFL" framework implements:
1. Data distribution across 5 simulated hospitals (non-IID fashion) with **data quality assessment**.
2. **Strategic selection** of 3 clients per round based on quality, history, and exploration.
3. Local client ANN model training with **DP-SGD via Opacus** (target ε≈5.0).
4. **Model improvement scores** calculated locally.
5. **Enhanced adaptive federated averaging** on the server using client data size, quality, and historical improvement.
6. Iteration for 10 communication rounds.
7. **Personalization** of the final global model for each client using fine-tuning and **Knowledge Distillation**.
8. Conceptual design for **layer-specific privacy sensitivity** in the ANN.

## Expected Outputs

The script generates and saves various visualizations and metrics in the `results/` folder:
- Global model performance (accuracy, F1 vs. rounds).
- Global model performance on each client's local data.
- Impact of personalization techniques (FT vs. KD).
- Privacy budget (ε) usage per hospital.
- Client selection frequency based on data quality.
- Analysis of data size, quality, contribution vs. performance.
- Privacy-Utility tradeoff visualization.
- Model robustness to input noise.
- Comparison with a centralized model.
- Feature importance analysis (optional, if enabled).
- CSV files for detailed metrics.

## Troubleshooting

- **Missing dependencies**: If you encounter `ModuleNotFoundError`, ensure all required packages are installed using `pip install -r requirements.txt`
- **CUDA issues**: If you encounter CUDA-related errors, try setting `device = 'cpu'` in the relevant code sections
- **Memory issues**: If you encounter memory problems, try reducing batch sizes or simplifying the model architecture

## Citation

If you use this code for your research, please cite:

```
@misc{federated_breast_cancer_diagnosis,
  author = {Kushagra Jaiswal},
  title = {Federated Deep Learning for Privacy-Preserving Breast Cancer Diagnosis},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/Nerd-coderZero/federated-breast-cancer-diagnosis}}
}
```

## License

[MIT License](LICENSE)
