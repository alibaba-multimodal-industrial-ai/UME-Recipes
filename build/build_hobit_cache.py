"""Build HOBIT's initial embeddings with the unmodified student checkpoint.

The underlying encoder path is shared with the already verified dynamic-batch
cache builder. This wrapper only gives the artifacts a HOBIT-specific manifest
owner so a teacher or unrelated cache cannot be accepted accidentally.
"""

import json
import os
from pathlib import Path

from build_dynamic_hard_batch_cache import main as build_pair_cache


def main():
    build_pair_cache()
    if int(os.environ.get("RANK", "0")) != 0:
        return
    manifest_path = Path(os.environ["DHBG_CACHE_DIR"]) / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("init_lora_model"):
        raise RuntimeError(
            "HOBIT initialization must use the student backbone, not a teacher adapter"
        )
    manifest["method"] = "hobit_initial_student_embeddings"
    manifest["hobit_cache_semantics"] = (
        "epoch0 checkpoint embeddings; replaced by rolling student forward cache"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[HOBIT_INITIAL_CACHE_OK] path={manifest_path.parent}", flush=True)


if __name__ == "__main__":
    main()
