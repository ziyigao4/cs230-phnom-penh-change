"""Generic SageMaker launcher.
usage: python3 launch_any.py NAME HOURS PROGRAM INSTANCE ch1=s3://... [ch2=s3://...] -- key=value ...
outputs written by the script to /opt/ml/checkpoints are synced to s3://BUCKET/runs/NAME/ckpt/
"""
import boto3, json, sys, time
B = 'cs230-ppchange-316936278582'; R = 'us-west-2'
ROLE = 'arn:aws:iam::316936278582:role/CS230SageMakerRole'
IMG = '763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.5.1-gpu-py311-cu124-ubuntu22.04-sagemaker'
name, hours, prog, inst = sys.argv[1], float(sys.argv[2]), sys.argv[3], sys.argv[4]
rest = sys.argv[5:]
cut = rest.index('--') if '--' in rest else len(rest)
chans = dict(a.split('=', 1) for a in rest[:cut]); extra = dict(a.split('=', 1) for a in rest[cut + 1:])
hp = {'sagemaker_program': prog, 'sagemaker_submit_directory': f's3://{B}/code/cs230_code.tar.gz',
      'sagemaker_region': R, 'sagemaker_container_log_level': 20, **extra}
hp = {k: json.dumps(v) for k, v in hp.items()}
job = f'{name}-{time.strftime("%m%d-%H%M%S")}'
boto3.client('sagemaker', region_name=R).create_training_job(
    TrainingJobName=job, RoleArn=ROLE, HyperParameters=hp,
    AlgorithmSpecification={'TrainingImage': IMG, 'TrainingInputMode': 'File'},
    InputDataConfig=[{'ChannelName': c, 'DataSource': {'S3DataSource': {'S3DataType': 'S3Prefix', 'S3Uri': u,
                      'S3DataDistributionType': 'FullyReplicated'}}} for c, u in chans.items()],
    OutputDataConfig={'S3OutputPath': f's3://{B}/output/'},
    ResourceConfig={'InstanceType': inst, 'InstanceCount': 1, 'VolumeSizeInGB': 100},
    StoppingCondition={'MaxRuntimeInSeconds': int(hours * 3600)},
    CheckpointConfig={'S3Uri': f's3://{B}/runs/{name}/ckpt/', 'LocalPath': '/opt/ml/checkpoints'})
print(job)
