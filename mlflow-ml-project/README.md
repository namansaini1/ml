# MLflow Machine Learning Project for arXiv Scientific Research Papers

This project utilizes the arXiv Scientific Research Papers dataset to build a machine learning model that can analyze and extract insights from research papers. The project is structured to facilitate data preprocessing, model training, evaluation, and logging using MLflow.

## Project Structure

```
mlflow-ml-project
├── data
│   └── arxiv_papers.csv          # Dataset containing arXiv research papers
├── notebooks
│   └── exploratory_data_analysis.ipynb  # Jupyter notebook for EDA
├── src
│   ├── data_preprocessing.py      # Data cleaning and preprocessing functions
│   ├── model_training.py           # Model training code with MLflow logging
│   ├── model_evaluation.py         # Model evaluation functions and logging
│   └── nlp_utils.py                # Utility functions for NLP tasks
├── models
│   └── model.pkl                   # Serialized trained model
├── experiments
│   └── mlruns                      # Directory for MLflow experiment logs
├── requirements.txt                # Python dependencies
├── MLproject                       # MLflow project definition
├── README.md                       # Project documentation
└── .gitignore                      # Files to ignore in Git
```

## Dataset

The dataset used in this project is `arxiv_papers.csv`, which contains various features such as:
- Title
- Abstract
- Authors
- Publication Date

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd mlflow-ml-project
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the exploratory data analysis notebook:
   ```
   jupyter notebook notebooks/exploratory_data_analysis.ipynb
   ```

## Usage

- **Data Preprocessing**: Use `src/data_preprocessing.py` to clean and preprocess the dataset.
- **Model Training**: Execute `src/model_training.py` to train the model and log the results with MLflow.
- **Model Evaluation**: Run `src/model_evaluation.py` to evaluate the trained model and log the evaluation metrics.

## MLflow Tracking

This project uses MLflow for tracking experiments. You can view the logged parameters, metrics, and models by running:
```
mlflow ui
```
Then navigate to `http://localhost:5000` in your web browser.

## License

This project is licensed under the MIT License. See the LICENSE file for details.