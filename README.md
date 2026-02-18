# Fake News Classifier 📰🔍

A production-ready NLP system that classifies news articles as **Fake** or **Real** using Machine Learning.

## 🚀 Features

- **NLP Preprocessing Pipeline**: Lowercasing, punctuation removal, stopword removal, and stemming.
- **TF-IDF Vectorization**: High-performance text-to-vector conversion.
- **Modular Architecture**: Separate modules for data loading, training, evaluation, and prediction.
- **REST API**: FastAPI-based server for real-time inference.
- **Configuration Driven**: All paths and parameters are managed via `config.yaml`.
- **Easy Deployment**: Includes a Dockerfile for containerization.

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd fake-news-classifier
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Training the Model

1. **Get the Data**:
   Place `Fake.csv` and `True.csv` from the [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) into the `data/` folder.

2. **Run Training**:
   ```bash
   python -m src.train
   ```
   This will:
   - Load and clean the data.
   - Train a Logistic Regression model.
   - Save the model and vectorizer to the `models/` directory.
   - Generate evaluation report and confusion matrix in `reports/`.

## 🌐 Running the API

Start the FastAPI server:
```bash
python -m api.app
```
The API will be available at `http://localhost:8000`. You can access the interactive Swagger documentation at `http://localhost:8000/docs`.

### Sample Curl Request
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application_json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "Breaking: A new study shows that drinking coffee makes you live forever!"
}'
```

## 🐳 Docker Usage

Build and run the container:
```bash
docker build -t fake-news-classifier .
docker run -p 8000:8000 fake-news-classifier
```

## 🏗️ Future Improvements

- [ ] Support for BERT or RoBERTa using Hugging Face Transformers.
- [ ] Integration with a database to log predictions.
- [ ] More robust unit tests for the preprocessing pipeline.
- [ ] Enhanced UI with news scraping capabilities.
