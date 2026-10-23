# Task 1: Multi-Class Benchmark Classification Pipeline (Wine)

End-to-end supervised classification on the Scikit-Learn Wine dataset (178 samples, 13 features, 3 classes).

## Pipeline
1. Load data and run exploratory pairplots (Seaborn)
2. 80/20 stratified train/test split
3. Feature scaling with `StandardScaler` (fit on train only to avoid leakage)
4. Train Logistic Regression and Decision Tree classifiers
5. Evaluate with accuracy, precision, recall, F1 and confusion matrices
6. Visualize decision boundaries on a 2D PCA projection

## Results
| Model | Test Accuracy |
|---|---|
| Logistic Regression | 97.2% |
| Decision Tree (max_depth=4) | 94.4% |

Full classification reports are in the notebook.

![Confusion matrices](images/confusion_matrices.png)
![Decision boundaries](images/decision_boundaries.png)

## Run it
```bash
pip install -r ../requirements.txt
jupyter notebook task1_wine_classification.ipynb
```

## Tools
Python, Pandas, NumPy, Scikit-Learn, Seaborn, Matplotlib
