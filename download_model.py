#!/usr/bin/env python3
"""Download the MIT-licensed TrackNet V1 tennis weights from Hugging Face."""

from huggingface_hub import hf_hub_download


if __name__ == "__main__":
    path = hf_hub_download(
        repo_id="vishnushenoy09/tracknet-v1-tennis",
        filename="tracknet_weights.pth",
        local_dir="models",
    )
    print(path)
