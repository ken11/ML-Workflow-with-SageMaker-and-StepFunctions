import boto3
from sagemaker.analytics import ExperimentAnalytics


def lambda_handler(event, context):
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
