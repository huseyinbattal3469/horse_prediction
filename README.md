# Horse Disease Outcome Prediction

This project is part of a Kaggle competition, [Playground Series - Season 3, Episode 22](https://www.kaggle.com/competitions/playground-series-s3e22), focusing on predicting horse disease outcomes. The project utilizes a variety of machine learning models, with a final ensemble approach to improve prediction accuracy.

## Project Files

- `train.csv`: Training dataset containing features and outcome labels.
- `test.csv`: Test dataset used for generating predictions.
- `required_functions.py`: Python script containing custom functions used for data preprocessing, outlier detection, scaling, encoding, and model optimization.
- `horse_result.csv`: The final prediction results generated by the model.

## Project Overview

The goal of this project is to predict the outcome of horse diseases based on a set of features. The project involves extensive exploratory data analysis (EDA), outlier detection and handling, missing value analysis, feature scaling, encoding, and model optimization using hyperparameter tuning.

### Data Preprocessing

- **Exploratory Data Analysis (EDA)**: The dataset is analyzed for both numerical and categorical features. Summary statistics are generated, and data distributions are visualized.
- **Outlier Detection and Handling**: Outliers are detected using box plots and are then handled by replacing values with thresholds to minimize their impact on the model.
- **Missing Value Analysis**: Missing values in the dataset are identified and appropriately handled.
- **Scaling and Encoding**: Numerical features are scaled using `RobustScaler`, and categorical features are encoded using `LabelEncoder`.

### Model Training and Optimization

- **Model Selection**: Multiple machine learning models are optimized using the `hyperparameter_multiclass_optimization` function, which performs hyperparameter tuning with cross-validation.
- **Ensemble Model**: A voting classifier is used to combine the best-performing models, resulting in a final ensemble model that is trained on the entire dataset.

### Prediction and Submission

- The final model is used to predict outcomes on the test dataset.
- The predictions are then saved in `horse_result.csv`, formatted for submission to the Kaggle competition.

## Installation

To run this project, ensure you have Python installed along with the required libraries. The necessary functions are provided in the `required_functions.py` file.

## Usage

1. **Run the Training Script**:
   ```bash
   python main.py
   ```
   This will preprocess the data, optimize the models, and generate predictions.

2. **Submit Predictions**:
   Upload `horse_result.csv` to the Kaggle competition to evaluate the model's performance.

## License

This project is created by Hüseyin Battal and is intended for educational purposes as part of the Kaggle competition.

GitHub: [https://github.com/huseyinbattal3469](https://github.com/huseyinbattal3469)  
LinkedIn: [https://www.linkedin.com/in/huseyin-battal/](https://www.linkedin.com/in/huseyin-battal/)
