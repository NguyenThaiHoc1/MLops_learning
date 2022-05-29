

import mlflow

if __name__ == '__main__':

    mlflow.set_experiment('ProjectFileExample')

    project_url = 'https://github.com/mlflow/mlflow-example'

    params = {'alpha': 0.5, 'l1_ratio': 0.01}

    mlflow.run(project_url, parameters=params)