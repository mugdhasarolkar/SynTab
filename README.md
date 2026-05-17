Synthetic Data Generator using Variational Autoencoder

A deep learning project that generates synthetic tabular data using Variational Autoencoders (VAE). The model learns patterns from real datasets and creates new synthetic samples with similar statistical properties.

Features
Synthetic tabular data generation
Custom Variational Autoencoder (VAE) implementation
TVAE (Tabular Variational Autoencoder) support
Streamlit-based web interface
CSV file upload support
Data preprocessing pipeline
Export generated synthetic data
Docker support for containerized deployment
Privacy-preserving synthetic data generation
Technologies Used
Python
TensorFlow / Keras
Streamlit
Pandas
NumPy
Scikit-learn
Matplotlib
Docker
Project Structure
Synthetic-Data-Generator/
│
├── models/
├── output/
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── generate.py
│   ├── vae.py
│   └── tvae.py
│
├── app.py
├── requirements.txt
├── Dockerfile
└── README.md
Installation

Clone the repository:

git clone https://github.com/your-username/synthetic-data-generator.git
cd synthetic-data-generator

Install dependencies:

pip install -r requirements.txt
Running the Project

Run the Streamlit application:

streamlit run app.py

The application allows users to directly upload CSV datasets through the file explorer interface and generate synthetic data.

Docker Support

Build Docker image:

docker build -t synthetic-data-generator .

Run Docker container:

docker run -p 8501:8501 synthetic-data-generator
Applications
Data augmentation
Privacy-preserving data sharing
Machine learning experiments
Handling limited datasets
Educational and research purposes
Future Improvements
Better support for categorical features
Enhanced latent space tuning
Improved synthetic data evaluation metrics
Cloud deployment integration
License

This project is licensed under the MIT License.
