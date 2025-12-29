from huggingface_hub import snapshot_download
from pathlib import Path

REPO_ID = "Maple728/Time-300B"
LOCAL_DIR = Path("./time300b_selected")

PARTITIONS = [
    "energy/electricity",
    "energy/energy_load",
    "finance/crypto_prices",
    "finance/stock_prices",
    "healthcare/hospital",
    "nature/sunspot",
    "other/m4_daily",
    "sales/favorita",
    "sales/dominick",
    "transport/traffic",
]

allow_patterns = [f"{p}/**" for p in PARTITIONS]

print("Downloading selected Time-300B partitions...")
snapshot_download(
    repo_id=REPO_ID,
    repo_type="dataset",
    local_dir=LOCAL_DIR,
    allow_patterns=allow_patterns,
    ignore_patterns=["**/*.lock"],
)

print("Download complete!")
