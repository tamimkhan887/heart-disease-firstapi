# Heart Disease Prediction API

A simple machine learning API built using FastAPI, Docker, and Random Forest for predicting the presence of heart disease.

## Project Objective

The goal of this project is to train a machine learning classifier using the Heart Disease Dataset, expose the model through a FastAPI REST API, containerize the application using Docker, and deploy it to Render.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Joblib
* FastAPI
* Pydantic
* Uvicorn
* Docker
* Docker Compose
* Render

## Project Structure

```text
heart-disease-api/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── model/
│   └── heart_model.joblib
│
├── train_model.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── .gitignore
└── README.md
```

## Machine Learning Model

The project uses a Random Forest Classifier.

The model uses the following features:

* age
* sex
* cp
* trestbps
* chol
* fbs
* restecg
* thalach
* exang
* oldpeak
* slope
* ca
* thal

The trained model is saved using Joblib:

```text
model/heart_model.joblib
```

## Running Locally

### 1. Create virtual environment

```bash
python -m venv venv
```

### 2. Activate virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

### 5. Start FastAPI

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

Swagger documentation:

```text
http://localhost:8000/docs
```

## API Endpoints

### GET /health

Checks whether the API is running.

Example response:

```json
{
  "status": "healthy"
}
```

### GET /info

Returns information about the model and its input features.

### POST /predict

Accepts patient feature values and returns a binary prediction.

Example request:

```json
{
  "age": 52,
  "sex": 1,
  "cp": 0,
  "trestbps": 125,
  "chol": 212,
  "fbs": 0,
  "restecg": 1,
  "thalach": 168,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 2,
  "ca": 2,
  "thal": 3
}
```

Example response:

```json
{
  "heart_disease": true
}
```

## Docker

Build the Docker image:

```bash
docker compose build
```

Run the application:

```bash
docker compose up
```

Run in background:

```bash
docker compose up -d
```

Stop the application:

```bash
docker compose down
```

Swagger:

```text
http://localhost:8000/docs
```

## Deployment

The application can be deployed to Render as a Docker Web Service.

Deployment steps:

1. Push the project to GitHub.
2. Create a new Web Service on Render.
3. Connect the GitHub repository.
4. Select Docker as the environment.
5. Use the project root as the Docker build context.
6. Deploy the service.
7. Test `/health`, `/info`, and `/predict`.

## Live Deployment

Render URL:

```text
https://heart-disease-firstapi.onrender.com/
```

Swagger:

```text
https://heart-disease-firstapi.onrender.com/docs
```

