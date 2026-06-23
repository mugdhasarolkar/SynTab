Generate synthetic tabular data from real CSV files using a Variational Autoencoder (VAE) and Tabular VAE (TVAE).
Train on your dataset and export new rows with similar patterns—useful for data augmentation, ML experiments, and privacy-friendly sharing.
## Features
- CSV upload and synthetic data export
- Custom VAE and TVAE models
- Data preprocessing pipeline
- Streamlit web UI
- Docker support
## Tech Stack
Python · TensorFlow/Keras · Streamlit · Pandas · NumPy · Scikit-learn · Matplotlib · Docker
## Project Structure
```
syntab/
├── app/
│   └── assets/              # UI assets
├── data/
│   ├── raw/                 # Input CSV files
│   └── New/                 # Generated synthetic data
├── models/
│   ├── train.py             # Train VAE/TVAE
│   ├── generate.py          # Generate synthetic samples
│   ├── vae_model.py         # VAE architecture
│   └── tvae.py              # Tabular VAE
├── preprocessing/
│   └── preprocess.py        # Data cleaning & encoding
├── evaluation/
│   ├── evaluate.py          # Quality metrics
│   └── plot.py              # Comparison plots
├── outputs/
│   └── saved_models/        # Trained models & preprocessors (.pkl)
├── notebooks/               # Experiments
├── app.py                   # Streamlit app
├── main.py                  # Main entry / pipeline
├── requirements.txt
└── dockerfile
```
## Installation
```bash
git clone https://github.com/mugdhasarolkar/SynTab.git
cd syntab
pip install -r requirements.txt
```
## Run
```bash
streamlit run app.py
```
Open **http://localhost:8501**, upload a CSV, train the model, and download synthetic data.
## Docker
```bash
docker build -t synthetic-app .
docker run -p 8501:8501 synthetic-app
```
## Use Cases
- Data augmentation  
- Privacy-preserving data sharing (review before production use)  
- ML experiments with limited data  
- Learning and research  
## Future Work
- Better categorical feature support  
- Latent space tuning  
- Synthetic data quality metrics  
## License
MIT License
