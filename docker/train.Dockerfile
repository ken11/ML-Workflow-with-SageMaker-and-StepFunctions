FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.4-cpu-py37

COPY train.py /opt/ml/code/train.py

ENV SAGEMAKER_PROGRAM train.py
