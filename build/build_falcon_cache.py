"""Build FALCON's epoch-0 rolling student representation cache.

Encoding is delegated to the already verified pair-cache builder.  This
wrapper changes only the manifest owner and records the exact FALCON cache
semantics, preventing accidental acceptance of teacher or HOBIT artifacts.
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
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("init_lora_model"):
        raise RuntimeError(
            "FALCON initialization must use the frozen student backbone, not "
            "a teacher adapter"
        )
    manifest["method"] = "falcon_initial_student_embeddings"
    manifest["falcon_cache_semantics"] = (
        "epoch-0 Qwen2-VL checkpoint embeddings; refreshed by each sampled "
        "training pair before its optimizer update"
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[FALCON_INITIAL_CACHE_OK] path={manifest_path.parent}", flush=True)


if __name__ == "__main__":
    main()
