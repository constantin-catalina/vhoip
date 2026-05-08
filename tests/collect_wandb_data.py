import wandb
import pandas as pd

api = wandb.Api()

# Replace with your entity and project name
runs = api.runs("constantin_catalina_viviana/vhoip")

all_data = []
for run in runs:
    # Skip deleted/crashed/unavailable runs
    if run.state not in ("finished", "running"):
        print(f"Skipping run '{run.name}' (state: {run.state})")
        continue

    try:
        history = run.history(samples=10000)
        if history.empty:
            print(f"Skipping run '{run.name}' — no history data")
            continue

        history["run_name"] = run.name
        history["run_id"]   = run.id
        history["state"]    = run.state
        # Optionally attach config params as columns
        for k, v in run.config.items():
            history[f"config/{k}"] = str(v)

        all_data.append(history)
        print(f"✓ Extracted run '{run.name}' ({len(history)} steps)")

    except Exception as e:
        print(f"Skipping run '{run.name}' — error: {e}")

if all_data:
    final_rows = []
    for i, history in enumerate(all_data):
        run_name = history["run_name"].iloc[0]
        separator = pd.DataFrame([{"run_name": f"=== {run_name} ==="}])
        final_rows.append(separator)
        final_rows.append(history)

    df = pd.concat(final_rows, ignore_index=True)
    df.to_csv("wandb_available_runs.csv", index=False)