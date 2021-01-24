# Heart Failure Prediction with Azure ML

This project demonstrates how to use an external dataset in a Microsoft Azure Machine Learning workspace; train a model using the different tools available in the AzureML framework as well deploy the model as a web service.

## Dataset

### Overview
To make predictions about heart failure, we used the open source [Heart Failure Prediction](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data/) dataset from kaggle  to build a classification model. This dataset uses 12 clinical features for predicting death events by heart failure.

### Task
Our goal is to find the best classification model for predicting death events by heart failure. We will build the models using hyperdrive and automl API from azureml. Given the 12 clinical features in the dataset our model will use DEATH_EVENT as the target column (binary: “1”, means “Yes”, “0” means “No”)

### Access
The dataset was registered into the workspace via URI of the public Heart Failure Prediction csv file.
![alt dataset registered](screenshots/1-dataset-registered.png)

## Automated ML
*TODO*: Give an overview of the `automl` settings and configuration you used for this experiment

### Results
*TODO*: What are the results you got with your automated ML model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Hyperparameter Tuning
*TODO*: What kind of model did you choose for this experiment and why? Give an overview of the types of parameters and their ranges used for the hyperparameter search


### Results
*TODO*: What are the results you got with your model? What were the parameters of the model? How could you have improved it?

*TODO* Remeber to provide screenshots of the `RunDetails` widget as well as a screenshot of the best model trained with it's parameters.

## Model Deployment
*TODO*: Give an overview of the deployed model and instructions on how to query the endpoint with a sample input.

## Screen Recording
[Screencast on YouTube](https://youtu.be/B8iwyPZ3j3k)

## Built With

* [Azure](https://portal.azure.com/) - Cloud service provider used
* [Python](https://www.python.org/) - The programming language used
* [Heart Failure Prediction dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv) - Dataset used

## Authors

* **[Pemberai Sweto](https://github.com/thepembeweb)** - *Initial work* - [Operationalizing Machine Learning pipeline in Azure
](https://github.com/thepembeweb/operationalizing_machine_learning_in_azure)

## License

[![License](http://img.shields.io/:license-mit-green.svg?style=flat-square)](http://badges.mit-license.org)

- This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details
- Copyright 2021 © [Pemberai Sweto](https://github.com/thepembeweb).
