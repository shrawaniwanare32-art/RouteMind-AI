#!/usr/bin/env bash
# One-time Azure setup: resource group, ACR, Container Apps environment + app,
# and the service principal GitHub Actions uses. Needs: `az login`. From the repo root:
#     bash deployment/azure/setup_azure.sh
set -euo pipefail

RG="${RG:-routemind-rg}"
LOCATION="${LOCATION:-centralindia}"
ACR="${ACR:-routemindacr${RANDOM}}"      # must be globally unique, lowercase letters/numbers only
ENV_NAME="routemind-env"
APP="routemind-api"
SUB_ID="$(az account show --query id -o tsv)"

az extension add --name containerapp --upgrade --yes >/dev/null
az provider register --namespace Microsoft.App --wait >/dev/null
az provider register --namespace Microsoft.OperationalInsights --wait >/dev/null

echo ">> Resource group + registry"
az group create --name "$RG" --location "$LOCATION" >/dev/null
az acr create --resource-group "$RG" --name "$ACR" --sku Basic --admin-enabled true >/dev/null

echo ">> Container Apps environment"
az containerapp env create --name "$ENV_NAME" --resource-group "$RG" --location "$LOCATION" >/dev/null

echo ">> Build the first image inside ACR (no local Docker needed)"
az acr build --registry "$ACR" --image "routemind-api:bootstrap" --file docker/Dockerfile.api .

echo ">> Create the container app"
ACR_USER="$(az acr credential show -n "$ACR" --query username -o tsv)"
ACR_PASS="$(az acr credential show -n "$ACR" --query 'passwords[0].value' -o tsv)"
az containerapp create --name "$APP" --resource-group "$RG" --environment "$ENV_NAME" \
  --image "${ACR}.azurecr.io/routemind-api:bootstrap" \
  --registry-server "${ACR}.azurecr.io" --registry-username "$ACR_USER" --registry-password "$ACR_PASS" \
  --target-port 8000 --ingress external --min-replicas 1 --max-replicas 2 \
  --cpu 0.5 --memory 1.0Gi >/dev/null

echo ">> Service principal for GitHub Actions"
CREDS="$(az ad sp create-for-rbac --name "routemind-github-${RANDOM}" --role contributor \
          --scopes "/subscriptions/${SUB_ID}/resourceGroups/${RG}" --sdk-auth)"

FQDN="$(az containerapp show -n "$APP" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
cat <<MSG

Done. API: https://${FQDN}/docs

In GitHub -> Settings -> Secrets and variables -> Actions add:
  secret:    AZURE_CREDENTIALS = (the JSON below - keep it private)
  variables: ACR_NAME=${ACR}   AZURE_RESOURCE_GROUP=${RG}   DEPLOY_AZURE=true

${CREDS}
MSG
