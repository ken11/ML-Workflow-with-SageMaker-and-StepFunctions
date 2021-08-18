FROM amazon/aws-lambda-python:3.7

COPY lambda_function.py ./
RUN pip install sagemaker boto3

CMD [ "lambda_function.lambda_handler" ]
