import os
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def evalute_metrcis(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    return rmse, mae


def run_training(**parameter):
    train_data = parameter['train_data']
    train_label = parameter['train_label']

    test_data = parameter['test_data']
    test_label = parameter['test_label']

    alpha = parameter['alpha']
    l1_ratio = parameter['l1_ratio']

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

    lr.fit(train_data, train_label)

    y_pred = lr.predict(test_data)

    rmse, mae = evalute_metrcis(test_label, y_pred)

    return rmse, mae


def mlfow_extend(mflow_path, **parameter):
    train_data = parameter['train_data']
    train_label = parameter['train_label']

    test_data = parameter['test_data']
    test_label = parameter['test_label']

    experiment_name = 'PlainRegression'

    try:
        exp_id = mlflow.create_experiment(name=experiment_name)
    except Exception as e:
        exp_id = mlflow.get_experiment_by_name(name=experiment_name).experiment_id

    mlflow_path_distribution = mflow_path / 'images' / 'distribution_plot_all_features.png'

    if 'images' not in os.listdir(mflow_path):
        os.mkdir('images')

    with mlflow.start_run(experiment_id=exp_id, run_name='Run1'):

        train_X.plot(kind='box', subplots=True, layout=(2, 4), title='Box plot of each feature')

        plt.savefig(str(mlflow_path_distribution))

        # training
        alpha, l1_ratio = 0.1, 0.05

        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)

        lr.fit(train_data, train_label)

        y_pred = lr.predict(test_data)

        rmse, mae = evalute_metrcis(test_label, y_pred)

        mlflow.log_param('alpha', alpha)
        mlflow.log_param('l1_ratio', l1_ratio)

        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)

        mlflow.log_artifacts('images')

        mlflow.sklearn.log_model(lr, 'PlainRegression_Model')


if __name__ == '__main__':
    root_project = Path(__file__).resolve().parent

    dataset_path = root_project / 'dataset' / 'auto_mpg' / 'auto-mpg.data'

    mlflow_tracker_path = root_project

    df = pd.read_csv(dataset_path,
                     names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                            "acceleration", "model_year", "origin", "car_name"],
                     delim_whitespace=True)

    for col in df.columns:
        if col not in ['mpg', 'car_name']:
            df = df[pd.to_numeric(df[col], errors='coerce').notnull()]

            df[col] = df[col].astype(float)

    X = df[["cylinders", "displacement", "horsepower", "weight",
            "acceleration", "model_year", "origin"]]

    y = df["mpg"]

    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=43)

    alphas, l1_ratios = [0.01, 0.02, 0.5], [0.15, 0.2, 0.5]

    # for alpha in alphas:
    #
    #     for l1_ratio in l1_ratios:
    #         rmse, mae = run_training(train_data=train_X, train_label=train_y,
    #                                  test_data=test_X, test_label=test_y,
    #                                  alpha=alpha, l1_ratio=l1_ratio)
    #
    #         print(f"Hyperparameters: Alpha = {alpha}, L1 Ratio = {l1_ratio} \n")
    #
    #         print(f"Model Performance on test set: RMSE {rmse}, MAE {mae} \n")
    #
    #         print("-" * 50, '\n')

    mlfow_extend(mlflow_tracker_path,
                 train_data=train_X, train_label=train_y,
                 test_data=test_X, test_label=test_y)
