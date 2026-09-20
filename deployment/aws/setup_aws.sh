#!/usr/bin/env bash
# One-time AWS setup: ECR repo, log group, ECS cluster + Fargate service.
# Needs: AWS CLI logged in (aws configure) and Docker running.  Run from the repo root:
#     bash deployment/aws/setup_aws.sh
set -euo pipefail

REGION="${AWS_REGION:-ap-south-1}"
APP="routemind"
ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"
REPO_URI="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${APP}-api"

echo ">> ECR repository"
aws ecr create-repository --repository-name "${APP}-api" --region "$REGION" \
  --image-scanning-configuration scanOnPush=true >/dev/null 2>&1 || echo "   (already exists)"

echo ">> CloudWatch log group"
aws logs create-log-group --log-group-name "/ecs/${APP}-api" --region "$REGION" 2>/dev/null || echo "   (already exists)"

echo ">> ECS task execution role"
aws iam get-role --role-name ecsTaskExecutionRole >/dev/null 2>&1 || {
  aws iam create-role --role-name ecsTaskExecutionRole --assume-role-policy-document '{
    "Version":"2012-10-17","Statement":[{"Effect":"Allow",
    "Principal":{"Service":"ecs-tasks.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name ecsTaskExecutionRole \
    --policy-arn arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
}

echo ">> Fill placeholders in task-definition.json"
sed -i.bak "s/REPLACE_ACCOUNT_ID/${ACCOUNT_ID}/g; s/REPLACE_REGION/${REGION}/g" deployment/aws/task-definition.json
rm -f deployment/aws/task-definition.json.bak

echo ">> Build + push the first image"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com"
docker build -f docker/Dockerfile.api -t "${REPO_URI}:latest" .
docker push "${REPO_URI}:latest"

echo ">> ECS cluster + task definition"
aws ecs create-cluster --cluster-name "${APP}-cluster" --region "$REGION" >/dev/null
aws ecs register-task-definition --cli-input-json file://deployment/aws/task-definition.json --region "$REGION" >/dev/null

echo ">> Networking (default VPC)"
VPC_ID="$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text --region "$REGION")"
SUBNETS="$(aws ec2 describe-subnets --filters Name=vpc-id,Values="$VPC_ID" --query 'Subnets[].SubnetId' --output text --region "$REGION" | tr '\t' ',')"
SG_ID="$(aws ec2 create-security-group --group-name "${APP}-sg" --description "RouteMind API" --vpc-id "$VPC_ID" --query GroupId --output text --region "$REGION")"
# DEMO ONLY: port 8000 open to the world. Put an ALB + HTTPS in front for anything real.
aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 8000 --cidr 0.0.0.0/0 --region "$REGION" >/dev/null

echo ">> ECS service"
aws ecs create-service --cluster "${APP}-cluster" --service-name "${APP}-api-service" \
  --task-definition "${APP}-api" --desired-count 1 --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[${SUBNETS}],securityGroups=[${SG_ID}],assignPublicIp=ENABLED}" \
  --region "$REGION" >/dev/null

cat <<MSG

Done. Now in GitHub -> Settings -> Secrets and variables -> Actions add:
  secrets:   AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY   (an IAM user limited to ECR + ECS)
  variables: AWS_REGION=${REGION}   DEPLOY_AWS=true
MSG
