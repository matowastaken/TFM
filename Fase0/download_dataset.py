import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

load_dotenv()

local_dir = snapshot_download(
    repo_id="waylonli/FINSABER-V2-Data",
    repo_type="dataset",
    local_dir=os.environ["FINSABER_DATA_PATH"],
)
print(local_dir)