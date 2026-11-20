"""Store the VLM API key without showing it: prompts with hidden input, writes ~/.config/aetos/vlm_api_key
(readable by you only). Run it in your own terminal:

    python scripts/set_vlm_key.py

scripts/plan_task_vlm.py reads the key from that file. Delete the file to remove the key.
"""
import getpass
import os
from pathlib import Path

path = Path.home() / ".config" / "aetos" / "vlm_api_key"
key = getpass.getpass("API key (input hidden): ").strip()
if not key:
    raise SystemExit("nothing entered; no file written")
path.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
with os.fdopen(fd, "w") as f:
    f.write(key + "\n")
os.chmod(path, 0o600)
print(f"saved to {path} ({len(key)} characters)")
