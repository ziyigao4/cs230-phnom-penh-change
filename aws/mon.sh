J=$1; N=${2:-25}
aws sagemaker describe-training-job --training-job-name $J --query '[TrainingJobStatus,SecondaryStatus,FailureReason,TrainingTimeInSeconds]' --output text
S=$(aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix $J --query 'logStreams[0].logStreamName' --output text 2>/dev/null)
[ "$S" != "None" ] && aws logs get-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name $S --limit $N --query 'events[].message' --output text | tr '\t' '\n' | cut -c1-200 | tail -$N
