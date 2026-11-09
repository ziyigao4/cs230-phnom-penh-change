#!/bin/bash
# pull latest results from the SageMaker checkpoint dir and rebuild RESULTS.md
set -e
B=${B:-cs230-ppchange-316936278582}; RUN=${RUN:-full}
cd "$(dirname "$0")"
aws s3 sync s3://$B/runs/$RUN/ckpt/ results/ --exclude '*.pt' --only-show-errors
aws s3 sync s3://$B/runs/$RUN/ckpt/ results/checkpoints/ --exclude '*' --include '*.pt' --only-show-errors
JOB=${JOB:-$(aws sagemaker list-training-jobs --name-contains $RUN- --sort-by CreationTime --sort-order Descending --max-results 1 --query "TrainingJobSummaries[0].TrainingJobName" --output text)}
aws logs filter-log-events --log-group-name /aws/sagemaker/TrainingJobs --log-stream-name-prefix $JOB --filter-pattern '?RESULT ?"===" ?" ep " ?FAILED ?patches ?"class counts"' --query "events[].message" --output text | tr "\t" "\n" > results/log.txt
python3 scripts/summarize.py results > /dev/null && echo synced
