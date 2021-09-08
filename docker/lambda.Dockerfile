FROM amazon/aws-lambda-python:3.7

RUN pip install sagemaker boto3

COPY lambda_function.py ./

CMD [ "lambda_function.lambda_handler" ]
