# Customer Service Intent Classifier
Mini project built during my internship at SURE ProEd (a unit of SURE Trust, Skill Upgradation for Rural-youth Empowerment).
Mentor: Mr. Vamsi Krishna | Author: Suraj Tiwari

A machine learning project that predicts the intent of a customer message using Natural Language Processing (NLP).

## Features

* Classifies customer messages into predefined intent classes.
* Supports an `out_of_scope` class for unrelated queries.
* Uses TF-IDF features with word and character n-grams.
* Uses Logistic Regression for classification.
* Includes typo augmentation to improve robustness to misspelled text.
* Saves the trained model pipeline for reuse.

## Tech Stack

* Python
* Pandas
* Scikit-learn
* TF-IDF
* Logistic Regression
* Joblib
* Streamlit

## Project Structure

```text
intent_app/
├── app.py
├── intent_pipeline.joblib
├── Intent_Classifier.ipynb
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── oos_train.txt
│   ├── oos_held.txt
│   └── README.md
└── venv/  # Local environment; not uploaded to GitHub
```

## Installation

1. Clone this repository:

   ```bash
   git clone (https://github.com/Suraj03-07/customer-service-intent-classifier.git)
   ```

2. Open the project directory:

   ```bash
   cd customer-service-intent-classifier
   ```

3. Create and activate a virtual environment.

   Windows:

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

4. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

## Run the Application

If the application uses Streamlit, run:

```bash
streamlit run app.py
```

## Model Training

The `Intent_Classifier.ipynb` notebook contains the data preprocessing, feature engineering, model training, and evaluation workflow.

The saved `intent_pipeline.joblib` file is used to make predictions without retraining the model.

## Evaluation

The training notebook reports:

* Validation accuracy: 99.88%
* Test accuracy: 99.88%
* Accuracy on synthetic typo examples: 99.27%

These results reflect the evaluation datasets used in the notebook. Performance on unseen real-world messages may differ.

## Limitations

* Predictions depend on the quality and coverage of the training data.
* The model may make incorrect predictions for ambiguous or unfamiliar messages.
* The confidence score should not be treated as a guarantee of correctness.

## Acknowledgements

SURE ProEd (a unit of SURE Trust) and my mentor Mr. Vamsi Krishna for guidance during the internship.

## Author

Suraj Tiwari
