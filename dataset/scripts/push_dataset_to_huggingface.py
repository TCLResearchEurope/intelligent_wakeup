"""
This code was developed by TCL Research Europe. For specific licensing terms, please refer to the
LICENSE file.

Script that upload corpus to hugging face.:
"""

import os
import shutil
import argparse
import pandas as pd
from huggingface_hub import HfApi, Repository


def validate_tsv(file_path: str):
    """
    Validates the structure of the TSV file to ensure it contains 'text' and 'label' columns.
    """
    df = pd.read_csv(file_path, sep="\t")
    required_columns = {"text", "label"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"The TSV file must contain the following columns: {required_columns}"
        )
    print(f"TSV file '{file_path}' is valid with {len(df)} rows.")
    return df


def upload_file_to_hf(
    file_path: str, repo_name: str, token: str, git_name: str, git_email: str
):
    """
    Uploads a TSV file to a Hugging Face dataset repository.
    """
    try:
        api = HfApi()
        # Specify repo_type="dataset" here
        repo_url = api.create_repo(
            repo_id=repo_name, token=token, exist_ok=True, repo_type="dataset"
        )

        repo_dir = f"./{repo_name}"
        if os.path.exists(repo_dir):
            shutil.rmtree(repo_dir)

        repo = Repository(
            local_dir=repo_dir,
            clone_from=repo_url,
            use_auth_token=token,
            git_user=git_name,
            git_email=git_email,
            repo_type="dataset",
        )

        dest_file_path = os.path.join(repo_dir, os.path.basename(file_path))
        shutil.copy2(file_path, dest_file_path)

        repo.git_add(pattern=".")
        repo.git_commit(f"Add dataset file: {os.path.basename(file_path)}")
        repo.git_push()
        print(f"Dataset '{file_path}' has been uploaded to {repo_url}")

    except Exception as e:
        print(f"Error occurred during upload: {str(e)}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push dataset to Hugging Face.")
    parser.add_argument("--file", required=True, help="Path to the TSV file to upload")
    parser.add_argument(
        "--repo_name", required=True, help="Hugging Face repository name"
    )
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument(
        "--git_name", required=True, help="Git username for the repository"
    )
    parser.add_argument(
        "--git_email", required=True, help="Git email for the repository"
    )

    args = parser.parse_args()
    validate_tsv(args.file)
    upload_file_to_hf(
        args.file, args.repo_name, args.token, args.git_name, args.git_email
    )
