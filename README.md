# ML Assignment 2 - Wall Following Robot Navigation

Submitted by: Mohamed Marzuk M, BITS ID: 2026AK05411

## a. Problem statement

A SCITOS G5 robot moves around a room by following the wall in the clockwise direction. It has 24 ultrasound sensors around its body. At each step it takes the 24 readings and then makes one of 4 moves: Move-Forward, Slight-Right-Turn, Sharp-Right-Turn or Slight-Left-Turn.

In this assignment I predict the move from the 24 sensor readings. I trained 5 classification models on the same data, compared them using 6 metrics, and made a Streamlit app where you can upload the test CSV and check the saved models.

## b. Dataset description

* Source: UCI Machine Learning Repository, Wall-Following Robot Navigation Data (Freire, Veloso and Barreto, 2009). I used the file `sensor_readings_24.data`. Link: https://archive.ics.uci.edu/dataset/194/wall+following+robot+navigation+data
* Rows: 5,456 (readings taken one after another while the robot was moving)
* Features: 24 numeric columns, US1 to US24. Each one is the distance in metres from one sensor. US1 is at the front and the sensors are 15 degrees apart. The maximum reading is about 5 m, which means the sensor did not detect anything.
* No missing values and no duplicate rows
* Target: `Movement` with 4 classes

| Class | Rows | Share |
| --- | --- | --- |
| Move-Forward | 2,205 | 40.4% |
| Sharp-Right-Turn | 2,097 | 38.4% |
| Slight-Right-Turn | 826 | 15.1% |
| Slight-Left-Turn | 328 | 6.0% |

Train/test split: the rows are in time order. The same label comes in small groups (396 groups, median size 6.5), and two rows next to each other are much more similar than two random rows (median distance 3.31 vs 8.66). If I shuffle and split randomly, the test set will have rows that are almost copies of training rows. So I did not shuffle. I used the first 80% of rows for training (4,364 rows) and the last 20% for testing (1,092 rows). This last 20% is saved as `test_data.csv`. All 4 classes are present in the test set (429 / 408 / 148 / 107 rows). Slight-Left-Turn is a little higher in the test set (9.8%) than in training (5.1%).

Preprocessing: mean imputation and StandardScaler. Both are inside the saved pipeline for each model, so the app can use the raw sensor columns directly.

## c. GitHub repository link

https://github.com/mohamedmarzukm-2026ak05411/ml_assignment

Live Streamlit app: https://mlassignment-ksamyoicjkigtx4xgbt9hf.streamlit.app/

```
project-folder/
|-- app.py
|-- requirements.txt
|-- README.md
|-- 2026AK05411_model_training.ipynb   training notebook (EDA, tuning, training, saves model/ and the csv files)
|-- test_data.csv            last 20% of the rows (24 sensors + Movement)
|-- model_metrics.csv
|-- split_comparison.csv     ordered vs random split results
|-- tuning_results.csv       best hyperparameters and CV scores
|-- model/
    |-- logistic_regression.joblib
    |-- decision_tree.joblib
    |-- knn.joblib
    |-- naive_bayes.joblib
    |-- svm.joblib
    |-- model_info.json
```

## d. Models used

All 5 models are trained on the same training rows and tested on the same 1,092 test rows. There are 4 classes, so I used macro average for Precision, Recall and F1, and one-vs-rest (macro) for AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.6658 | 0.9142 | 0.6509 | 0.7535 | 0.6612 | 0.5572 |
| Decision Tree | 0.9817 | 0.9949 | 0.9737 | 0.9872 | 0.9800 | 0.9734 |
| KNN | 0.8059 | 0.9366 | 0.7905 | 0.8103 | 0.7973 | 0.7197 |
| Naive Bayes | 0.5385 | 0.8645 | 0.5765 | 0.6797 | 0.5511 | 0.4308 |
| Support Vector Machine | 0.8947 | 0.9758 | 0.8949 | 0.8971 | 0.8950 | 0.8460 |

Hyperparameter tuning: I used GridSearchCV on the training data only, with TimeSeriesSplit (4 folds, gap = 10) so that the validation rows always come after the training rows. The gap leaves out 10 rows between the training and validation part of each fold, because rows next to each other are almost the same reading. The score used was macro F1. I took 4 folds because with 5 folds the first training fold has no Slight-Left-Turn rows. The test data was used only once at the end, for the scores in the table above. For Logistic Regression, Decision Tree and SVM I set class_weight = balanced because Slight-Left-Turn is only 6% of the data. random_state = 16.

| ML Model Name | Values tried | Best values | CV macro F1 |
| --- | --- | --- | --- |
| Logistic Regression | C: 0.01, 0.1, 1, 10, 100 | C = 10 | 0.5676 |
| Decision Tree | max_depth: 4 to 12, min_samples_leaf: 1, 5, 10, 20 | max_depth = 10, min_samples_leaf = 1 | 0.9476 |
| KNN | k: 3 to 21, weights: uniform / distance | k = 5, distance | 0.6878 |
| Naive Bayes | var_smoothing: 1e-9 to 1e-1 | 1e-3 | 0.5386 |
| Support Vector Machine | C: 0.3 to 100, gamma: scale, 0.01, 0.03, 0.1 | C = 100, gamma = 0.03 | 0.7929 |

For SVM the best C (100) is the highest value in my grid. But going from C = 30 to C = 100 only increased the CV score from 0.7867 to 0.7929, so I did not try higher values.

I also ran the same tuned models on a normal random split (random_state = 16) to compare with my ordered split:

| ML Model Name | Accuracy random | Accuracy ordered | MCC random | MCC ordered |
| --- | --- | --- | --- | --- |
| Logistic Regression | 0.6859 | 0.6658 | 0.5711 | 0.5572 |
| Decision Tree | 0.9954 | 0.9817 | 0.9931 | 0.9734 |
| KNN | 0.8947 | 0.8059 | 0.8409 | 0.7197 |
| Naive Bayes | 0.5348 | 0.5385 | 0.3985 | 0.4308 |
| Support Vector Machine | 0.9341 | 0.8947 | 0.9002 | 0.8460 |

### Observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Accuracy is only 0.666 but AUC is 0.914. So it ranks the classes okay, but a linear boundary is not enough to separate them. Because of class_weight it predicts Slight-Left-Turn a lot: recall is 0.95 but precision is only 0.46. It finds only about half of the Move-Forward rows (recall 0.52). |
| Decision Tree | Best model on all 6 metrics (accuracy 0.982, MCC 0.973). I think this is because the robot decides its move using distance rules (if a sensor is closer than some value then turn), and a decision tree learns the same type of if-else rules. Lowest F1 is for Slight-Right-Turn (0.95). Accuracy dropped only a little from the random split to the ordered split (0.995 to 0.982), much less than KNN, so it is not just memorising nearby rows. |
| KNN | Accuracy 0.806, MCC 0.720. This model dropped the most when I changed from the random split to the ordered split (accuracy 0.895 to 0.806). With a random split, the nearest neighbour of a test row is usually the reading just before or after it, so the score was too high. Move-Forward has the lowest recall (0.72). |
| Naive Bayes | Lowest scores of all models (accuracy 0.539, MCC 0.431). Naive Bayes assumes every sensor is independent and normally distributed. That is not true here: neighbouring sensors are correlated (median 0.40) and about 14% of readings are stuck at 5 m. It predicts the slight turns too many times, for example Slight-Right-Turn precision is only 0.33. |
| Support Vector Machine | Second best (accuracy 0.895, AUC 0.976, MCC 0.846). Tuning made the biggest difference for SVM, accuracy went from 0.820 (C = 1) to 0.895 (C = 100, gamma = 0.03). The RBF kernel can make non-linear boundaries, so it does much better than Logistic Regression. Most of its mistakes are on Move-Forward rows (recall 0.84). Accuracy dropped about 0.04 with the ordered split. |
| Overall, Winner for your dataset? | Decision Tree is the best model for this dataset. It is best on all 6 metrics and its rules are easy to explain. SVM is second but about 0.09 lower in accuracy. The random vs ordered split comparison also showed that KNN and SVM look better than they really are when the data is shuffled. |

## Streamlit app features

* Upload the test CSV (there is also a button to download the sample `test_data.csv`)
* Dropdown to choose the model
* Accuracy, AUC, Precision, Recall, F1 and MCC for the chosen model
* Confusion matrix, ROC curves and classification report
* Table and bar chart comparing all 5 models on the uploaded file
* Polar plot of the 24 sensor readings for one row, with the actual and predicted move
* Download the predictions as CSV

The app only loads the saved `.joblib` models and predicts. It does not train again on the uploaded file.

## How to run locally

```
pip install -r requirements.txt
streamlit run app.py
```

To train the models again, run `2026AK05411_model_training.ipynb` from top to bottom. It downloads the dataset from UCI and writes the models and csv files to an `output/` folder. I ran it on BITS Virtual Lab (Python 3.12).

The models were saved with scikit-learn 1.7.2, so the same version is used in requirements.txt. On Streamlit Community Cloud I selected Python 3.14 in Advanced settings.
