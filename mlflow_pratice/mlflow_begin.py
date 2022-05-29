import os
from random import randint, random

import matplotlib
import mlflow
import pandas as pd
import sklearn

print("MLflow Version: ", mlflow.version.VERSION)
print("Pandas Version: ", pd.__version__)
print("Scikit-learn Version: ", sklearn.__version__)
print("Matplotlib Version: ", matplotlib.__version__)


def run(run_name=""):
    mlflow.set_experiment("helloWorld")

    with mlflow.start_run() as r:
        print("Running helloWorld.ipynb")

        print("Model run: ", r.info.run_uuid)

        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.log_param("param1", randint(0, 100))

        mlflow.log_metric("foo", random())
        mlflow.log_metric("foo1", random() + 1)

        mlflow.set_tag("run_origin", "Jupyter_notebook")

        if not os.path.exists("outputs"):
            os.makedirs("outputs")

        with open("outputs/test.txt", "w") as f:
            f.write("hello world !")

        mlflow.log_artifacts("outputs", artifact_path="artifact")
        mlflow.end_run()


run("LocalRun")
