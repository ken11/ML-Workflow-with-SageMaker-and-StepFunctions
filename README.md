# ML Workflow with AWS SageMaker and StepFunctions
## About
I referred to [this notebook](https://github.com/aws-samples/amazon-sagemaker-examples-jp/blob/master/step-functions-data-science-sdk/model-train-evaluate-compare/step_functions_mlworkflow_scikit_learn_data_processing_and_model_evaluation_with_experiments.ipynb).  
The contents of this repository make it easy to create ML Workflow without using a notebook.  
As an example, run the MNIST preprocessing, training, and model evaluation processes in StepFunctions.  
![example](https://user-images.githubusercontent.com/2043460/129913250-403b6a1a-40e6-4590-8c9f-2dcb6d5d0430.png)

## How to use
### 1. Prepare the AWS environment
#### a. Create S3 Bucket
- Create an S3 bucket for use with Workflow.

#### b. Create ECR Repository
- Use container image in SageMaker Processing. Create an ECR Repository to store this image.

#### c. Create execution role
To execute it, you need two roles, the SageMaker execution role and the Step Functions Workflow execution role.
- __SageMaker execution role__:  
  Create a role that has access to the created S3 and ECR. Also touch the policies of SageMaker and StepFunctions to this role.
- __Workflow execution role__ :  
  Create a Workflow execution role for Step Functions according to [this](https://sagemaker-examples.readthedocs.io/en/latest/step-functions-data-science-sdk/step_functions_mlworkflow_processing/step_functions_mlworkflow_scikit_learn_data_processing_and_model_evaluation.html#Create-an-Execution-Role-for-Step-Functions) explanation.

### 2. Upload docker image and input data, create source.tar.gz
#### a. Build and upload docker image
```sh
$ docker build -t your-ecr-repo-name -f docker/Dockerfile .
# If you want to use your own image instead of the built-in image
$ docker build -t your-train-ecr-repo-name -f docker/train.Dockerfile .
$ docker build -t your-lambda-ecr-repo-name -f docker/lambda.Dockerfile .
```
Push the built image to the ECR repository.  
This image is used to run SageMaker Processing.

#### b. Prepare and upload input data
Download the MNIST data from [here](http://yann.lecun.com/exdb/mnist/).  
There are 4 gz files.  
- t10k-images-idx3-ubyte.gz
- t10k-labels-idx1-ubyte.gz
- train-images-idx3-ubyte.gz
- train-labels-idx1-ubyte.gz

Make this a single zip file named `input.zip`.
```sh
$ zip input.zip t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz
```
Upload the zip file to S3.

#### c. Create source.tar.gz
```sh
$ tar zcvf source.tar.gz train.py
```

### 3. Create lambda function
Create a lambda function using the container image for lambda that you uploaded earlier.  
Assign a policy that can access SageMaker and S3 to the execution role of the lambda function.


### 4. Configuration
Edit `config.yml`.
```yaml
aws:
  role:
    sagemaker_execution: Set the ARN for the SageMaker execution role you created in the previous step.
    workflow_execution: Set the ARN for the Workflow execution role you created in the previous step.
  bucket: Your S3 Bucket name
  ecr_repository_uri: Set the URI of the Docker image you uploaded in the previous step.
  input_data_s3_uri: Set the URI of `input.zip` that you uploaded to S3 in the previous step.
stepfunctions:
  workflow:
    name:
sagemaker:
  experiment:
    name:
    bucket_name: Bucket where you want to store the CSV of the Experiment
    key: Key when saving CSV of Experiment
  processing:
    preprocess:
      job_name_prefix:
      instance_count:
      instance_type: execution instance type such as ml.m5.xlarge
      max_runtime_in_seconds:
    evaluation:
      job_name_prefix:
      instance_count:
      instance_type:
      max_runtime_in_seconds:
  training:
    job_name_prefix:
    instance_count:
    instance_type:
    use_spot_instances:
    max_run:
    max_wait:
    hyperparameters:
      learning_rate: '0.001'
      epochs: '5'
    image_uri: If you want to use your own image instead of the built-in image
lambda:
  function_name: The name of the lambda function you just created
```
See the [official documentation](https://sagemaker.readthedocs.io/en/stable/index.html) for details on options such as instance type.

### 5. Execution
```sh
# Don't forget to set the environment in docker-compose.yml.
$ docker-compose run --rm app bash
$ python workflow.py
```

### (Option) 6. QuickSight
![experiment](https://user-images.githubusercontent.com/2043460/129915623-20952ace-7221-4b83-9044-7ada4f754f52.png)
Visualize the results of the Experiment with QuickSight. Since the CSV of Experiment is stored in the S3 bucket specified in config, specify it in the dataset of QuickSight.
![qs](https://user-images.githubusercontent.com/2043460/129914164-4dd43c33-c914-46f7-9e56-409bbd39d0c9.png)
The following is a sample manifest.json when specifying in the QuickSight dataset.
```json
{
    "fileLocations": [
        {
            "URIs": [
                "s3://**********/*********.csv"
            ]
        },
        {
            "URIPrefixes": [
            ]
        }
    ],
    "globalUploadSettings": {
        "format": "CSV",
        "delimiter": ",",
        "textqualifier": "'",
        "containsHeader": "true"
    }
}
```

## Option
You can specify the path of the `config.yml` file.
```sh
$ python workflow.py -c your-config-file-path
```
