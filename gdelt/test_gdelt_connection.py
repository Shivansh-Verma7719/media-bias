import argparse
import sys
from google.cloud import bigquery
from google.oauth2 import service_account

def make_bq_client(gcp_project, key_file=None):
    """
    Creates a BigQuery client.
    Uses a service account JSON key if provided, otherwise falls back to
    Application Default Credentials (ADC).
    """
    if key_file:
        try:
            credentials = service_account.Credentials.from_service_account_file(
                key_file,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            return bigquery.Client(project=gcp_project, credentials=credentials)
        except Exception as e:
            print(f"❌ Error loading service account key: {e}")
            sys.exit(1)
    
    # Fallback to Application Default Credentials
    return bigquery.Client(project=gcp_project)

def test_connection():
    parser = argparse.ArgumentParser(description="Test GDELT BigQuery Connection")
    parser.add_argument("--gcp-project", default="media-bias-ism", help="GCP project ID")
    parser.add_argument("--key-file", help="Path to service account JSON key file")
    args = parser.parse_args()

    print(f"🔍 Testing connection to GCP Project: {args.gcp_project}...")
    if args.key_file:
        print(f"🔑 Using service account key: {args.key_file}")
    else:
        print("🌍 Using Application Default Credentials (ADC)")

    try:
        client = make_bq_client(args.gcp_project, args.key_file)
        
        # Simple query: Count GDELT GKG entries for a single day.
        # This scans very little data (~100MB) due to partitioning.
        query = """
            SELECT COUNT(*) as row_count
            FROM `gdelt-bq.gdeltv2.gkg_partitioned`
            WHERE _PARTITIONTIME = TIMESTAMP("2024-01-01")
        """
        
        print("🚀 Running test query against gdelt-bq.gdeltv2.gkg_partitioned...")
        job = client.query(query)
        result = list(job.result(timeout=30))
        
        count = result[0].row_count
        print(f"\n✅ Connection Successful!")
        print(f"📊 GKG entries found on 2024-01-01: {count:,}")
        print("\nYour credentials have permission to query the public GDELT dataset.")

    except Exception as e:
        print(f"\n❌ Connection Failed!")
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print("1. Ensure BigQuery API is enabled in your GCP project.")
        print("2. Verify your service account has 'BigQuery User' or 'BigQuery Job User' role.")
        print("3. Check if the path to your JSON key file is correct.")
        if not args.key_file:
            print("4. If not using a key file, ensure you ran 'gcloud auth application-default login'.")
        sys.exit(1)

if __name__ == "__main__":
    test_connection()