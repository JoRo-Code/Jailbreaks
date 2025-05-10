import os
import requests
import logging

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

SECRET = os.environ.get("LAMBDA_API_KEY", None)
assert SECRET is not None, "LAMBDA_API_KEY is not set"

def get_instance_ids():
    url = "https://cloud.lambda.ai/api/v1/instances"
    response = requests.get(url, auth=(SECRET, ''))
    return [instance['id'] for instance in response.json()['data']]

def terminate_instances(instance_ids):
    url = "https://cloud.lambda.ai/api/v1/instance-operations/terminate"
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json"
    }

    response = requests.post(
        url,
        auth=(SECRET, ''),
        json={"instance_ids": instance_ids},
        headers=headers,
        timeout=30,
    )

    response.raise_for_status()
    return response.json()

if __name__ == "__main__":
    ids = get_instance_ids()
    logger.info(f"Active instances: {ids}")
    logger.info(f"Terminating {len(ids)} instances")
    logger.info(terminate_instances(ids))

