from datetime import datetime, timedelta, timezone

import argparse
import boto3
import sagemaker
import stepfunctions
import yaml
import time
from sagemaker.processing import (ProcessingInput, ProcessingOutput,
                                  ScriptProcessor)
from sagemaker.tensorflow.estimator import TensorFlow
from sagemaker.estimator import Estimator
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps import Chain, ProcessingStep
from stepfunctions.steps.states import Retry, Choice, Fail, Catch
from stepfunctions.steps.choice_rule import ChoiceRule
from stepfunctions.workflow import Workflow


class MLWorkflow:

    def __init__(self, conf):
        self.workflow_name = conf['stepfunctions']['workflow']['name']
        self.sagemaker_session = sagemaker.Session()
        self.region = self.sagemaker_session.boto_region_name

        self.sagemaker_execution_role = conf['aws']['role']['sagemaker_execution']
        self.workflow_execution_role = conf['aws']['role']['workflow_execution']

        self.account_id = boto3.client('sts').get_caller_identity().get('Account')
        self.bucket = conf['aws']['bucket']
        self.repository_uri = conf['aws']['ecr_repository_uri']

        self.data_update = conf['sagemaker']['data_update']
        self.preprocess_job_name_prefix = conf['sagemaker']['processing']['preprocess']['job_name_prefix']
        self.train_job_name_prefix = conf['sagemaker']['training']['job_name_prefix']
        self.evaluation_job_name_prefix = conf['sagemaker']['processing']['evaluation']['job_name_prefix']

        self.experiment_name = conf['sagemaker']['experiment']['name']
        self.experiment_bucket_name = conf['sagemaker']['experiment']['bucket_name']
        self.experiment_key = conf['sagemaker']['experiment']['key']
        JST = timezone(timedelta(hours=+9), 'JST')
        self.timestamp = datetime.now(JST).strftime("%Y-%m-%d-%H-%M-%S")
        self.hyperparameters = conf['sagemaker']['training']['hyperparameters']
        self.learning_rate = self.hyperparameters['learning_rate']
        self.epochs = self.hyperparameters['epochs']

        self.preprocessor_settings = conf['sagemaker']['processing']['preprocess']
        self.estimator_settings = conf['sagemaker']['training']
        self.evaluation_processor_settings = conf['sagemaker']['processing']['evaluation']
        self.lambda_settings = conf['lambda']

        self.input = conf['aws']['input_data_s3_uri']

        self.execution_input = ExecutionInput(
            schema={
                "DataUpdate": bool,
                "PreprocessingJobName": str,
                "PreprocessingInputData": str,
                "PreprocessingOutputDataTrain": str,
                "PreprocessingOutputDataTest": str,
                "TrainingJobName": str,
                "TrainingParameters": dict,
                "TrainingOutputModel": str,
                "ExperimentName": str,
                "EvaluationProcessingJobName": str,
                "EvaluationProcessingOutput": str,
            }
        )

        self.experiment = self._create_experiments(self.experiment_name)
        self.trial = self._create_trial(self.experiment_name)

    # Workflow creation
    def create(self):
        preprocess_job_name, train_job_name, evaluation_job_name = self._create_job_name()
        s3_bucket_base_uri = f"s3://{self.bucket}"
        output_data = f"{s3_bucket_base_uri}/data/processing/output-{self.timestamp}"
        model_data_s3_uri = f"{s3_bucket_base_uri}/{train_job_name}/output/model.tar.gz"
        output_model_evaluation_s3_uri = f"{s3_bucket_base_uri}/{train_job_name}/evaluation"

        # Creating each step
        data_source_step = self._data_source()
        preprocess_step = self._preprocess()
        train_step, train_code = self._train(f"{s3_bucket_base_uri}/{train_job_name}/output", data_source_step)
        evaluation_step = self._evaluation(train_step)
        experiment_upload_step = self._experiment_upload()

        # Determine whether to execute preprocess.
        # If there is a data update, preprocessing is executed.
        # (Judged by the contents of the `DataUpdate` key of ExecutionInput)
        choice_state = Choice("Determine whether to execute preprocess.")
        choice_state.add_choice(
            rule=ChoiceRule.BooleanEquals(variable="$.DataUpdate", value=True),
            next_step=preprocess_step
        )
        choice_state.add_choice(
            rule=ChoiceRule.BooleanEquals(variable="$.DataUpdate", value=False),
            next_step=data_source_step
        )

        # Create a step when it fails
        failed_state_sagemaker_processing_failure = Fail(
            "ML Workflow failed", cause="SageMakerProcessingJobFailed"
        )
        catch_state_processing = Catch(
            error_equals=["States.TaskFailed"],
            next_step=failed_state_sagemaker_processing_failure,
        )
        data_source_step.add_catch(catch_state_processing)
        preprocess_step.add_catch(catch_state_processing)
        train_step.add_catch(catch_state_processing)
        evaluation_step.add_catch(catch_state_processing)
        experiment_upload_step.add_catch(catch_state_processing)

        # execution
        workflow_graph = Chain([choice_state, preprocess_step, data_source_step, train_step, evaluation_step, experiment_upload_step])
        branching_workflow = Workflow(
            name=self.workflow_name,
            definition=workflow_graph,
            role=self.workflow_execution_role,
        )
        branching_workflow.create()
        branching_workflow.update(workflow_graph)

        # NOTE: The update will not be reflected immediately, so you have to wait for a while.
        time.sleep(5)

        branching_workflow.execute(
            inputs={
                "DataUpdate": self.data_update,
                "PreprocessingJobName": preprocess_job_name,
                "PreprocessingInputData": self.input,
                "PreprocessingOutputDataTrain": output_data + '/train_data',
                "PreprocessingOutputDataTest": output_data + '/test_data',
                "TrainingJobName": train_job_name,
                "TrainingParameters": {
                    "sagemaker_program": "train.py",
                    "sagemaker_submit_directory": train_code,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs
                },
                "TrainingOutputModel": model_data_s3_uri,
                "ExperimentName": self.experiment.experiment_name,
                "EvaluationProcessingJobName": evaluation_job_name,
                "EvaluationProcessingOutput": output_model_evaluation_s3_uri
            }
        )

    # Select a data source according to whether the data has been updated.
    # If the data has not been updated, select the latest preprocessed data from the past Experiments data.
    def _data_source(self):
        step = stepfunctions.steps.compute.LambdaStep(
            "data source",
            parameters={
                "FunctionName": self.lambda_settings['data_source']['function_name'],
                "Payload": {
                    "StateInput.$": "$",
                    "data_update": self.execution_input["DataUpdate"],
                    "experiment-name": self.experiment_name,
                    "bucket_name": self.bucket,
                    "job": "data_source"
                },
            },
        )
        step.add_retry(
            Retry(error_equals=["States.TaskFailed"], interval_seconds=15, max_attempts=2, backoff_rate=4.0)
        )
        return step

    # pre-process step creation
    def _preprocess(self):
        # https://sagemaker.readthedocs.io/en/stable/api/training/processing.html?highlight=ScriptProcessor#sagemaker.processing.ScriptProcessor
        preprocessor = ScriptProcessor(
            command=['python3'],
            image_uri=self.repository_uri,
            role=self.sagemaker_execution_role,
            sagemaker_session=self.sagemaker_session,
            instance_count=self.preprocessor_settings['instance_count'],
            instance_type=self.preprocessor_settings['instance_type'],
            max_runtime_in_seconds=self.preprocessor_settings['max_runtime_in_seconds']
        )
        input_code = self._upload('preprocess.py', "data/preprocess/code")

        # Define inputs and outputs
        inputs = [
            ProcessingInput(
                source=self.execution_input["PreprocessingInputData"], destination="/opt/ml/processing/input", input_name="source_input"
            ),
            ProcessingInput(
                source=input_code,
                destination="/opt/ml/processing/input/code",
                input_name="preprocess_code",
            ),
        ]
        outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/train",
                destination=self.execution_input["PreprocessingOutputDataTrain"],
                output_name="train_data",
            ),
            ProcessingOutput(
                source="/opt/ml/processing/test",
                destination=self.execution_input["PreprocessingOutputDataTest"],
                output_name="test_data",
            ),
        ]

        # https://aws-step-functions-data-science-sdk.readthedocs.io/en/v2.1.0/sagemaker.html?highlight=ProcessingStep#stepfunctions.steps.sagemaker.ProcessingStep
        return ProcessingStep(
            "Preprocessing step",
            processor=preprocessor,
            job_name=self.execution_input["PreprocessingJobName"],
            inputs=inputs,
            outputs=outputs,
            experiment_config={
                "ExperimentName": self.execution_input["ExperimentName"],
                'TrialName': self.trial.trial_name,
                'TrialComponentDisplayName': 'Preprocess'
            },
            container_entrypoint=[
                "python3", "/opt/ml/processing/input/code/preprocess.py"
            ],
        )

    # training step creation
    def _train(self, model_dir, step):
        # NOTE: max_wait can be specified only when using spot instance.
        if self.estimator_settings['use_spot_instances']:
            max_wait = self.estimator_settings['max_wait']
        else:
            max_wait = None

        # NOTE: You can also use your own container image.
        if self.estimator_settings['image_uri']:
            estimator = Estimator(
                image_uri=self.estimator_settings['image_uri'],
                instance_count=self.estimator_settings['instance_count'],
                instance_type=self.estimator_settings['instance_type'],
                use_spot_instances=self.estimator_settings['use_spot_instances'],
                max_run=self.estimator_settings['max_run'],
                max_wait=max_wait,
                role=self.sagemaker_execution_role,
                output_path=f"s3://{self.bucket}",
                metric_definitions=[
                    {'Name': 'train:loss', 'Regex': '.*?loss: (.*?) -'},
                    {'Name': 'train:accuracy', 'Regex': '.*?accuracy: (0.\\d+).*?'},
                ],
            )
        else:
            # https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html#tensorflow-estimator
            estimator = TensorFlow(
                entry_point="train.py",
                instance_count=self.estimator_settings['instance_count'],
                instance_type=self.estimator_settings['instance_type'],
                use_spot_instances=self.estimator_settings['use_spot_instances'],
                max_run=self.estimator_settings['max_run'],
                max_wait=max_wait,
                role=self.sagemaker_execution_role,
                framework_version="2.4",
                py_version="py37",
                output_path=f"s3://{self.bucket}",
                metric_definitions=[
                    {'Name': 'train:loss', 'Regex': '.*?loss: (.*?) -'},
                    {'Name': 'train:accuracy', 'Regex': '.*?accuracy: (0.\\d+).*?'},
                ],
            )
        train_code = self._upload('source.tar.gz', 'data/train/code')

        # https://aws-step-functions-data-science-sdk.readthedocs.io/en/v2.1.0/sagemaker.html?highlight=ProcessingStep#stepfunctions.steps.sagemaker.TrainingStep
        return steps.TrainingStep(
            "Training Step",
            estimator=estimator,
            data={"train_data": sagemaker.TrainingInput(
                step.output()["Payload"]["train_data"], content_type="application/octet-stream"
            )},
            job_name=self.execution_input["TrainingJobName"],
            experiment_config={
                "ExperimentName": self.execution_input["ExperimentName"],
                'TrialName': self.trial.trial_name,
                'TrialComponentDisplayName': 'Train'
            },
            hyperparameters=self.execution_input["TrainingParameters"],
            wait_for_completion=True,
            result_path="$.TrainResult"
        ), train_code

    # evaluation step creation
    def _evaluation(self, step):
        evaluation_processor = ScriptProcessor(
            command=['python3'],
            image_uri=self.repository_uri,
            role=self.sagemaker_execution_role,
            sagemaker_session=self.sagemaker_session,
            instance_count=self.evaluation_processor_settings['instance_count'],
            instance_type=self.evaluation_processor_settings['instance_type'],
            max_runtime_in_seconds=self.evaluation_processor_settings['max_runtime_in_seconds']
        )
        evaluation_code = self._upload('evaluation.py', "data/evaluation/code")
        inputs = [
            ProcessingInput(
                source=step.output()["Payload"]["test_data"],
                destination="/opt/ml/processing/test",
                input_name="test_data",
            ),
            ProcessingInput(
                source=self.execution_input["TrainingOutputModel"],
                destination="/opt/ml/processing/model",
                input_name="model",
            ),
            ProcessingInput(
                source=evaluation_code,
                destination="/opt/ml/processing/input/code",
                input_name="evaluation_code",
            ),
        ]
        outputs = [
            ProcessingOutput(
                source="/opt/ml/processing/evaluation",
                destination=self.execution_input["EvaluationProcessingOutput"],
                output_name="evaluation",
            ),
        ]

        return ProcessingStep(
            "Evaluation step",
            processor=evaluation_processor,
            job_name=self.execution_input["EvaluationProcessingJobName"],
            inputs=inputs,
            outputs=outputs,
            experiment_config={
                "ExperimentName": self.execution_input["ExperimentName"],
                'TrialName': self.trial.trial_name,
                'TrialComponentDisplayName': 'Evaluation'
            },
            container_entrypoint=["python3", "/opt/ml/processing/input/code/evaluation.py"],
        )

    def _experiment_upload(self):
        step = stepfunctions.steps.compute.LambdaStep(
            "Upload Experiment",
            parameters={
                "FunctionName": self.lambda_settings['experiments']['function_name'],
                "Payload": {
                    "experiment-name": self.experiment_name,
                    "experiment_bucket_name": self.experiment_bucket_name,
                    "experiment_key": self.experiment_key,
                    "job": "experiment_upload"
                },
            },
        )
        step.add_retry(
            Retry(error_equals=["States.TaskFailed"], interval_seconds=15, max_attempts=2, backoff_rate=4.0)
        )
        return step

    def _create_job_name(self):
        preprocess_job_name = f"{self.preprocess_job_name_prefix}-{self.timestamp}"
        train_job_name = f"{self.train_job_name_prefix}-{self.timestamp}"
        evaluation_job_name = f"{self.evaluation_job_name_prefix}-{self.timestamp}"
        return preprocess_job_name, train_job_name, evaluation_job_name

    def _upload(self, file, prefix):
        return self.sagemaker_session.upload_data(
            file,
            bucket=self.bucket,
            key_prefix=prefix,
        )

    def _create_experiments(self, experiment_name):
        try:
            experiment = Experiment.load(experiment_name=experiment_name)
        except Exception as ex:
            if "ResourceNotFound" in str(ex):
                experiment = Experiment.create(
                    experiment_name=experiment_name,
                    description="example project experiments",
                    sagemaker_boto_client=boto3.client('sagemaker'))

        return experiment

    def _create_trial(self, experiment_name):
        return Trial.create(
            trial_name=self.timestamp,
            experiment_name=self.experiment_name,
            sagemaker_boto_client=boto3.client('sagemaker'),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default='config.yml')
    args, _ = parser.parse_known_args()

    with open(args.c, 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.SafeLoader)

    MLWorkflow(conf).create()
