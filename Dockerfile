FROM python:3.8-slim

RUN apt-get update && apt-get install -y gcc wget vim less git

RUN pip install --upgrade pip
# There is a bug in stepfunctions v2.2.0 and hyperparameters cannot be set.
# https://github.com/aws/aws-step-functions-data-science-sdk-python/issues/152
RUN pip install awscli boto3 sagemaker "stepfunctions==2.1.0" sagemaker-experiments

WORKDIR "/work"
