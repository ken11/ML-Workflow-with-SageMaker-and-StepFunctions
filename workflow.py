from datetime import datetime, timedelta, timezone

import argparse
import boto3
import sagemaker
import stepfunctions
import yaml
from sagemaker.processing import (ProcessingInput, ProcessingOutput,
                                  ScriptProcessor)
from sagemaker.tensorflow.estimator import TensorFlow
from smexperiments.experiment import Experiment
from stepfunctions import steps
from stepfunctions.inputs import ExecutionInput
from stepfunctions.steps import Chain, ProcessingStep
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

        self.preprocess_job_name_prefix = conf['sagemaker']['processing']['preprocess']['job_name_prefix']
        self.train_job_name_prefix = conf['sagemaker']['training']['job_name_prefix']
        self.evaluation_job_name_prefix = conf['sagemaker']['processing']['evaluation']['job_name_prefix']

        self.experiment_name = conf['sagemaker']['experiment']['name']
        self.hyperparameters = conf['sagemaker']['training']['hyperparameters']
        self.learning_rate = self.hyperparameters['learning_rate']

        self.preprocessor_settings = conf['sagemaker']['processing']['preprocess']
        self.estimator_settings = conf['sagemaker']['training']
        self.evaluation_processor_settings = conf['sagemaker']['processing']['evaluation']

        self.input = conf['aws']['input_data_s3_uri']

        self.execution_input = ExecutionInput(
            schema={
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
                "EvaluationExperimentArgs": list,
            }
        )

        self.experiment_evaluate = self._create_experiments(self.experiment_name)

    # Workflow creation
    def create(self):
        preprocess_job_name, train_job_name, evaluation_job_name, timestamp = self._create_job_name()
        s3_bucket_base_uri = f"s3://{self.bucket}"
        output_data = f"{s3_bucket_base_uri}/data/processing/output-{timestamp}"
        model_data_s3_uri = f"{s3_bucket_base_uri}/{train_job_name}/output/model.tar.gz"
        output_model_evaluation_s3_uri = f"{s3_bucket_base_uri}/{train_job_name}/evaluation"

        # Creating each step
        preprocess_step = self._preprocess()
        train_step, train_code = self._train(f"{s3_bucket_base_uri}/{train_job_name}/output")
        evaluation_step = self._evaluation()

        # Create a step when it fails
        failed_state_sagemaker_processing_failure = stepfunctions.steps.states.Fail(
            "ML Workflow failed", cause="SageMakerProcessingJobFailed"
        )
        catch_state_processing = stepfunctions.steps.states.Catch(
            error_equals=["States.TaskFailed"],
            next_step=failed_state_sagemaker_processing_failure,
        )
        preprocess_step.add_catch(catch_state_processing)
        train_step.add_catch(catch_state_processing)
        evaluation_step.add_catch(catch_state_processing)

        # execution
        workflow_graph = Chain([preprocess_step, train_step, evaluation_step])
        branching_workflow = Workflow(
            name=self.workflow_name,
            definition=workflow_graph,
            role=self.workflow_execution_role,
        )
        branching_workflow.create()
        branching_workflow.update(workflow_graph)

        branching_workflow.execute(
            inputs={
                "PreprocessingJobName": preprocess_job_name,
                "PreprocessingInputData": self.input,
                "PreprocessingOutputDataTrain": output_data + '/train_data',
                "PreprocessingOutputDataTest": output_data + '/test_data',
                "TrainingJobName": train_job_name,
                "TrainingParameters": {
                    "sagemaker_program": "train.py",
                    "sagemaker_submit_directory": train_code,
                    "lr": self.learning_rate,
                },
                "TrainingOutputModel": model_data_s3_uri,
                "ExperimentName": self.experiment_evaluate.experiment_name,
                "EvaluationProcessingJobName": evaluation_job_name,
                "EvaluationProcessingOutput": output_model_evaluation_s3_uri,
                "EvaluationExperimentArgs": ['--experiment-name', self.experiment_evaluate.experiment_name]
            }
        )

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
                source=self.execution_input["PreprocessingInputData"], destination="/opt/ml/processing/input", input_name="input-1"
            ),
            ProcessingInput(
                source=input_code,
                destination="/opt/ml/processing/input/code",
                input_name="code",
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
            container_entrypoint=[
                "python3", "/opt/ml/processing/input/code/preprocess.py"
            ],
        )

    # training step creation
    def _train(self, model_dir):
        # NOTE: max_wait can be specified only when using spot instance.
        if self.estimator_settings['use_spot_instances']:
            max_wait = self.max_wait
        else:
            max_wait = None

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
            output_path=f"s3://{self.bucket}"
        )
        train_code = self._upload('source.tar.gz', 'data/train/code')

        # https://aws-step-functions-data-science-sdk.readthedocs.io/en/v2.1.0/sagemaker.html?highlight=ProcessingStep#stepfunctions.steps.sagemaker.TrainingStep
        return steps.TrainingStep(
            "Training Step",
            estimator=estimator,
            data={"train": sagemaker.TrainingInput(
                self.execution_input["PreprocessingOutputDataTrain"], content_type="application/octet-stream"
            )},
            job_name=self.execution_input["TrainingJobName"],
            hyperparameters=self.execution_input["TrainingParameters"],
            wait_for_completion=True,
        ), train_code

    # evaluation step creation
    def _evaluation(self):
        evaluation_processor = ScriptProcessor(
            command=['python3'],
            image_uri=self.repository_uri,
            role=self.sagemaker_execution_role,
            sagemaker_session=self.sagemaker_session,
            instance_count=1,
            instance_type='ml.m5.xlarge',
            max_runtime_in_seconds=600
        )
        evaluation_code = self._upload('evaluation.py', "data/evaluation/code")
        inputs = [
            ProcessingInput(
                source=self.execution_input["PreprocessingOutputDataTest"],
                destination="/opt/ml/processing/test",
                input_name="input-1",
            ),
            ProcessingInput(
                source=self.execution_input["TrainingOutputModel"],
                destination="/opt/ml/processing/model",
                input_name="input-2",
            ),
            ProcessingInput(
                source=evaluation_code,
                destination="/opt/ml/processing/input/code",
                input_name="code",
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
            experiment_config={"ExperimentName": self.execution_input["ExperimentName"]},
            container_arguments=self.execution_input["EvaluationExperimentArgs"],
            container_entrypoint=["python3", "/opt/ml/processing/input/code/evaluation.py"],
        )

    def _create_job_name(self):
        JST = timezone(timedelta(hours=+9), 'JST')
        timestamp = datetime.now(JST).strftime("%Y-%m-%d-%H-%M-%S")
        preprocess_job_name = f"{self.preprocess_job_name_prefix}-{timestamp}"
        train_job_name = f"{self.train_job_name_prefix}-{timestamp}"
        evaluation_job_name = f"{self.evaluation_job_name_prefix}-{timestamp}"
        return preprocess_job_name, train_job_name, evaluation_job_name, timestamp

    def _upload(self, file, prefix):
        return self.sagemaker_session.upload_data(
            file,
            bucket=self.bucket,
            key_prefix=prefix,
        )

    def _create_experiments(self, experiment_name):
        try:
            experiment_evaluate = Experiment.load(experiment_name=experiment_name)
        except Exception as ex:
            if "ResourceNotFound" in str(ex):
                experiment_evaluate = Experiment.create(
                    experiment_name=experiment_name,
                    description="model evaluation",
                    sagemaker_boto_client=boto3.client('sagemaker'))

        return experiment_evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", type=str, default='config.yml')
    args, _ = parser.parse_known_args()

    with open(args.c, 'rb') as f:
        conf = yaml.load(f.read(), Loader=yaml.SafeLoader)

    MLWorkflow(conf).create()
