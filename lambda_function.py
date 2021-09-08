import boto3
from sagemaker.analytics import ExperimentAnalytics


def lambda_handler(event, context):
    if event['job'] == 'data_source':
        return data_source(event)
    elif event['job'] == 'experiment_upload':
        experiment_upload(event)


def experiment_upload(event):
    experiment_name = event['experiment-name']
    bucket_name = event['experiment_bucket_name']
    key = event['experiment_key']
    trial_component_analytics = ExperimentAnalytics(
        experiment_name=experiment_name,
        input_artifact_names=[]
    )
    df = trial_component_analytics.dataframe()
    s3 = boto3.resource('s3')
    s3_obj = s3.Object(bucket_name, key)
    s3_obj.put(Body=df.to_csv(None).encode('utf_8_sig'))


def data_source(event):
    data_update = event['data_update']
    experiment_name = event['experiment-name']
    bucket_name = event['bucket_name']
    # @TODO
    # using variable for DesplayName
    search_expression = {
        "Filters": [
            {
                "Name": "DisplayName",
                "Operator": "Contains",
                "Value": "Preprocess",
            },
        ],
    }

    trial_component_analytics = ExperimentAnalytics(
        experiment_name=experiment_name,
        search_expression=search_expression,
        input_artifact_names=[]
    )
    df = trial_component_analytics.dataframe()
    df = df[df['train_data - Value'].notnull()]

    if data_update is False:
        client = boto3.client('s3')
        for _, row in df.iterrows():
            key = row['train_data - Value'].split(f"{bucket_name}/")[1] + '/'
            result = client.list_objects(Bucket=bucket_name, Prefix=key)
            if "Contents" in result:
                train_data = row['train_data - Value']
                test_data = row['test_data - Value']
                break
    else:
        train_data = event['StateInput']["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
        test_data = event['StateInput']["ProcessingOutputConfig"]["Outputs"][1]["S3Output"]["S3Uri"]
    return {"train_data": train_data, "test_data": test_data}
