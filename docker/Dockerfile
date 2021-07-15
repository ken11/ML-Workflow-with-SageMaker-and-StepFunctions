FROM python:3.7-slim-buster
    
ENV AWS_DEFAULT_REGION us-west-2

RUN pip3 install --upgrade pip
RUN pip3 install -qU boto3 pandas==0.25.3 sagemaker-experiments sagemaker tensorflow numpy
