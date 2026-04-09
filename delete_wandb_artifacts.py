import os
from pathlib import Path

import wandb


def load_env_file(path: str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

load_env_file()
PROJECT = os.getenv("WANDB_PROJECT")

if not PROJECT:
    raise ValueError("Missing WANDB_PROJECT in .env")

api = wandb.Api()

for artifact_type in api.artifact_types(project=PROJECT):
    for collection in artifact_type.collections():
        versions = api.artifacts(artifact_type.type, f"{PROJECT}/{collection.name}")
        for version in versions:
            try:
                version.delete(delete_aliases=True)
                print(f"  Deleted: {version.name} v{version.version}")
            except Exception as e:
                print(f"  Failed:  {version.name} v{version.version} — {e}")

print("Done.")
