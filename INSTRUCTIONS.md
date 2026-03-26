# Google Cloud & DVC Setup Instructions

**1. Log in to your personal Google Cloud account.**
This allows your terminal to authenticate and talk directly to Google Cloud APIs.
```bash
gcloud auth login
```

**2. Set your active Google Cloud project.**
This tells `gcloud` which project (like an overarching workspace) all of the following commands should apply to.
```bash
gcloud config set project practice-ml-project-402717 
```

**3. Create a Google Cloud Storage bucket.**
This establishes a secure, remote folder in the cloud where your DVC data and Vertex AI machine learning artifacts will be saved.
```bash
gcloud storage buckets create gs://dogs-vs-cats-bucket \
    --default-storage-class=STANDARD \
    --location=EUROPE-WEST6 \
    --uniform-bucket-level-access \
    --public-access-prevention
```

**4. Enable required Google Cloud APIs.**
This flips the switch in GCP to allow you to use Virtual Machines (Compute), Cloud Storage, and Vertex AI (for machine learning jobs).
```bash
gcloud services enable compute.googleapis.com \
                       storage.googleapis.com \
                       aiplatform.googleapis.com
```

**5. Grant Vertex AI permissions to the Service Account.**
This gives your custom `pipeline-runner` robotic account the ability to submit, run, and manage AI training jobs on Vertex AI.
```bash
gcloud projects add-iam-policy-binding practice-ml-project-402717 \
    --member="serviceAccount:pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

**6. Grant Storage permissions to the Service Account.**
This gives the robotic account the ability to read your datasets and write its finished models directly into your Cloud Storage bucket.
```bash
gcloud projects add-iam-policy-binding practice-ml-project-402717 \
    --member="serviceAccount:pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
    --role="roles/storage.admin"
```

**7. Grant Service Account "Act As" permissions on itself.**
When submitting a Vertex AI job, Google Cloud strictly requires the submitting identity to possess the "Service Account User" role to execute jobs under its own name.
```bash
gcloud projects add-iam-policy-binding practice-ml-project-402717 \
    --member="serviceAccount:pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
    --role="roles/iam.serviceAccountUser"
```

**8. Configure local DVC to use the Google Cloud bucket.**
This tells the local DVC tool on your laptop to push and pull your large datasets to a `/dvc-store` subfolder securely hidden in your new GCS Bucket.
```bash
source .venv/bin/activate
dvc remote add -d gcs-remote gs://dogs-vs-cats-bucket/dvc-store
```

**9. Enable IAM Credentials for GitHub Actions.**
This allows external systems (like GitHub Actions) to generate short-lived, secure login tokens into Google Cloud without needing dangerous, permanent password keys.
```bash
gcloud services enable iamcredentials.googleapis.com \
    --project="practice-ml-project-402717"
```

**10. Create a Workload Identity Pool for GitHub.**
Think of an identity "pool" as an exclusive VIP guest list for external services. We are creating one specifically for GitHub Actions runners.
```bash
gcloud iam workload-identity-pools create "github-pool" \
  --project="practice-ml-project-402717" \
  --location="global" \
  --display-name="GitHub Actions Pool"
```

**11. Connect GitHub to the VIP Pool (OIDC Provider).**
This securely pairs GitHub's authentication system with Google Cloud, strictly limiting access *only* to code originating from your specific `lucaswerner90/test_on_gcp` code repository.
```bash
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="practice-ml-project-402717" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Auth Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository" \
  --attribute-condition="assertion.repository == 'lucaswerner90/test_on_gcp'" \
  --issuer-uri="https://token.actions.githubusercontent.com"
```

**12. Allow GitHub Actions to "become" the pipeline-runner.**
Finally, this grants GitHub Actions (arriving securely through the VIP pool) the legal permission to impersonate your `pipeline-runner` Service Account and deploy the actual ML code.
```bash
gcloud iam service-accounts add-iam-policy-binding "pipeline-runner@practice-ml-project-402717.iam.gserviceaccount.com" \
  --project="practice-ml-project-402717" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/426929788583/locations/global/workloadIdentityPools/github-pool/attribute.repository/lucaswerner90/test_on_gcp"
```