"""
    https://www.youtube.com/watch?v=WbicniUy_u0
"""
import os
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


def read_csv(path_csv):
    df = pd.read_csv(path_csv, sep=',', header=None,
                     names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "evaluation"])

    return df


def mlflow_extend(path_mlflow_sys, **kwargs):
    train_x = kwargs['train_x']
    train_y = kwargs['train_y']

    test_x = kwargs['test_x']
    test_y = kwargs['test_y']
    experiment_name = 'SimpleClassification'

    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except:
        exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    if 'data_processed' not in os.listdir(path=path_mlflow_sys):
        os.mkdir('data_processed')

    with mlflow.start_run(experiment_id=exp_id, run_name='Run_CarClassification'):

        mlflow.set_tag('Description', 'Simple Classification Model')
        mlflow.set_tags({
            'ProblemType': 'Classification',
            'ModelType': 'DecisionTree',
            'ModelLibrary': 'Scikit-learn',
        })

        encoder = OneHotEncoder(handle_unknown='ignore')

        X_encoded_train = encoder.fit_transform(train_x)
        train_x_encoded = pd.DataFrame(X_encoded_train.toarray())

        X_encoded_test = encoder.transform(test_x)
        test_x_encoded = pd.DataFrame(X_encoded_test.toarray())

        train_x_encoded.to_csv('data_processed/encoded_train.csv', sep='|', index=False)
        test_x_encoded.to_csv('data_processed/encoded_test.csv', sep='|', index=False)

        max_depth, max_features = 5, 11
        clf = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features, random_state=42)

        clf.fit(train_x_encoded, train_y)

        clf.predict(test_x_encoded)

        accuracy = clf.score(test_x_encoded, test_y)

        # setting mlflow attribution
        mlflow.log_artifacts('data_processed')

        mlflow.shap.log_explanation(clf.predict_proba, train_x_encoded)
        mlflow.log_param('max_depth', max_depth)
        mlflow.log_param('max_features', max_features)

        mlflow.log_metric('accuracy', accuracy)

        mlflow.sklearn.log_model(clf, 'SimpleClassification_Model')


if __name__ == '__main__':
    root_project = Path(__file__).resolve().parent

    dataset_path = root_project / 'dataset' / 'car_evolution' / 'car.data'

    df = read_csv(path_csv=dataset_path)

    X = df[["buying", "maint", "doors", "persons", "lug_boot", "safety"]]

    y = df["evaluation"]

    train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=43)

    mlflow_extend(path_mlflow_sys=root_project,
                  train_x=train_x, train_y=train_y,
                  test_x=test_x, test_y=test_y)
