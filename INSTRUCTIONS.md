```bash
gcloud auth login
```

```bash
gcloud config set project practice-ml-project-402717 
```

```bash
gcloud storage buckets create gs://dogs-vs-cats-bucket \
    --default-storage-class=STANDARD \
    --location=EUROPE-WEST6 \
    --uniform-bucket-level-access \
    --public-access-prevention
```

```bash
gcloud services enable compute.googleapis.com \
                       storage.googleapis.com \
                       aiplatform.googleapis.com
```

```bash
gcloud projects add-iam-policy-binding practice-ml-project-402717 \
    --member="serviceAccount:pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

```bash
gcloud projects add-iam-policy-binding practice-ml-project-402717 \
    --member="serviceAccount:pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

```bash
gcloud projects add-iam-policy-binding practice-ml-project-402717 \
    --member="serviceAccount:pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

```bash
source .venv/bin/activate
dvc remote add -d gcs-remote gs://dogs-vs-cats-bucket/dvc-store
```

```bash
# Enable IAM Credentials API
gcloud services enable iamcredentials.googleapis.com \
    --project="practice-ml-project-402717"

# Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --project="practice-ml-project-402717" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create GitHub OIDC Provider with restrictve repository condition
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="practice-ml-project-402717" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Auth Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == 'lucaswerner90/test_on_gcp'" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Bind Service Account to the Workload Identity Pool specifically for your GitHub repository
gcloud iam service-accounts add-iam-policy-binding "pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
  --project="practice-ml-project-402717" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/426929788583/locations/global/workloadIdentityPools/github-pool/attribute.repository/lucaswerner90/test_on_gcp"
```