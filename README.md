# COMP473-Project

## Description
This project implements and compares different fine-tuning approaches for DistilBERT on the IMDB sentiment analysis task, including:
- Standard fine-tuning with various layer freezing strategies
- LoRA (Low-Rank Adaptation) fine-tuning
- DoRA (Weight-Decomposed Low-Rank Adaptation) fine-tuning

## Requirements

### Python Libraries
The project requires the following Python packages:

- **torch** - PyTorch deep learning framework
- **transformers** - Hugging Face transformers library for pre-trained models
- **datasets** - Hugging Face datasets library for loading IMDB dataset
- **pandas** - Data manipulation and CSV handling
- **numpy** - Numerical computing
- **scikit-learn** - Machine learning metrics (accuracy, F1 score, confusion matrix)
- **tqdm** - Progress bars for training loops

### Hardware Requirements
- **GPU recommended** - CUDA-capable GPU for faster training (code includes CUDA support)
- CPU training is supported but will be significantly slower

## Installation

### Windows Command Prompt
```cmd
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pandas numpy scikit-learn tqdm
```

### Conda (Windows/Linux/Mac)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c huggingface transformers datasets
conda install pandas numpy scikit-learn tqdm
```

### Linux (Ubuntu/Debian)
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets pandas numpy scikit-learn tqdm
```

### Alternative: Install from requirements.txt
You can also create a `requirements.txt` file with the following content:
```
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
pandas>=1.5.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

Then install using:
```bash
# Windows CMD or Linux
pip install -r requirements.txt

# Conda
conda install --file requirements.txt
```

## Usage

### 1. Data Preparation
The notebooks will automatically download the IMDB dataset from Hugging Face and split it into train/validation/test sets. The data will be saved to CSV files in the `data/` directory.

### 2. Running the Notebooks
The project includes three main Jupyter notebooks:

- `fine-tune-with-layers.ipynb` - Fine-tuning with different layer freezing strategies
- `fine-tune-with-LoRA.ipynb` - Fine-tuning with LoRA adaptation
- `fine-tune-with-DoRA.ipynb` - Fine-tuning with DoRA adaptation

Run the notebooks in Jupyter Lab or VS Code to train the models.

### 3. Pre-trained Models
The repository includes pre-trained model checkpoints:
- `best_model_DistilBERT_classifier.pt`
- `best_model_DistilBERT_DoRA.pt`
- `best_model_DistilBERT_LoRA.pt`
- `best_model_DistilBERT_none.pt`
- `best_model_DistilBERT_top_layers.pt`

## Project Structure
```
COMP473-Project/
├── data/                           # Dataset directory
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
├── fine-tune-with-layers.ipynb     # Layer freezing experiments
├── fine-tune-with-LoRA.ipynb       # LoRA fine-tuning
├── fine-tune-with-DoRA.ipynb       # DoRA fine-tuning
├── best_model_*.pt                 # Pre-trained model checkpoints
├── performances.txt                # Model performance results
└── README.md                       # This file
```

## Notes
- The code automatically detects and uses CUDA if a GPU is available
- Mixed precision training (AMP) is enabled for faster training on compatible GPUs
- The notebooks include grid search for hyperparameter optimization
- Training progress is displayed with tqdm progress bars

