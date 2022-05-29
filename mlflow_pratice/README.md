## 1. Comparing runs
Run `mlflow ui` in a terminal or `http://your-tracking-server-host:5000` to view the experiment log and visualize and compare different runs and experiments. The logs and the model artifacts are saved in the mlruns directory (or where you specified).

## 2. Packaging the experiment as a MLflow project as conda env
Specify the entrypoint for this project by creating a MLproject file and adding an conda environment with a conda.yaml. You can copy the yaml file from the experiment logs.

To run this project, invoke `mlflow run . -P alpha=0.42`. After running this command, MLflow runs your training code in a new Conda environment with the dependencies specified in conda.yaml.

## 3. Deploy the model
Deploy the model locally by running

`mlflow models serve -m mlruns/0/f5f7c052ddc5469a852aa52c14cabdf1/artifacts/model/ -h 0.0.0.0 -p 1234`

Test the endpoint:

`curl -X POST -H "Content-Type:application/json; format=pandas-split" --data '{"columns":["alcohol", "chlorides", "citric acid", "density", "fixed acidity", "free sulfur dioxide", "pH", "residual sugar", "sulphates", "total sulfur dioxide", "volatile acidity"],"data":[[12.8, 0.029, 0.48, 0.98, 6.2, 29, 3.33, 1.2, 0.39, 75, 0.66]]}' http://0.0.0.0:1234/invocations`

You can also simply build a docker image from your model

`mlflow models build-docker -m mlruns/1/d671f37a9c7f478989e67eb4ff4d1dac/artifacts/model/ -n elastic_net_wine`

and run the container with

`docker run -p 8080:8080 elastic_net_wine.`

Or you can directly deploy to AWS sagemaker or Microsoft Azure ML.