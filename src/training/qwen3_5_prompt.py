"""Isolated, role-aware prompt profiles for Qwen3.5 embedding experiments.

The default profile is intentionally a no-op.  ``embedding_v2`` is enabled
only by the dedicated train/eval scripts and uses the same registry in both
paths, preventing prompt drift between training and MMEB evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Optional


LEGACY_PROFILE = "legacy"
EMBEDDING_V2_PROFILE = "embedding_v2"
PROMPT_CONFIG_NAME = "qwen3_5_prompt_config.json"
PROMPT_IMPLEMENTATION = (
    "qwen3.5-role-aware-embedding-prompt-v2.1-exact-payload-cleaning"
)


def _key(name: str) -> str:
    name = (name or "").split("/")[-1].strip().lower()
    return re.sub(r"[^a-z0-9]+", "", name)


_TASK_DATASETS = {
    "classification": {
        "ImageNet-1K", "ImageNet_1K", "N24News", "HatefulMemes",
        "SUN397", "VOC2007", "Place365", "ImageNet-A", "ImageNet-R",
        "ObjectNet", "Country211",
    },
    "question_answering": {
        "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA",
        "Visual7W", "ScienceQA", "GQA", "TextVQA", "VizWiz",
    },
    "image_to_text": {"VisualNews_i2t", "MSCOCO_i2t"},
    "text_to_image": {
        "VisualNews_t2i", "MSCOCO_t2i", "VisDial", "WebQA", "EDIS",
        "Wiki-SS-NQ",
    },
    "composed_image_retrieval": {"CIRR", "FashionIQ"},
    "visual_grounding": {
        "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing",
    },
    "image_retrieval": {"NIGHTS", "OVEN"},
}
TASK_BY_DATASET = {
    _key(dataset): task
    for task, datasets in _TASK_DATASETS.items()
    for dataset in datasets
}


PROMPTS = {
    ("classification", "query"): (
        "Encode the image as a single dense vector for matching its class "
        "label. The final token is used as the embedding."
    ),
    ("classification", "target"): (
        "Encode the class label as a single dense vector for matching an "
        "image. The final token is used as the embedding."
    ),
    ("question_answering", "query"): (
        "Encode the image and question as a single dense vector for matching "
        "the correct answer. The final token is used as the embedding."
    ),
    ("question_answering", "target"): (
        "Encode the candidate answer as a single dense vector for matching an "
        "image-question pair. The final token is used as the embedding."
    ),
    ("image_to_text", "query"): (
        "Encode the image as a single dense vector for retrieving its matching "
        "caption. The final token is used as the embedding."
    ),
    ("image_to_text", "target"): (
        "Encode the caption as a single dense vector for retrieving its "
        "matching image. The final token is used as the embedding."
    ),
    ("text_to_image", "query"): (
        "Encode the text as a single dense vector for retrieving its matching "
        "image. The final token is used as the embedding."
    ),
    ("text_to_image", "target"): (
        "Encode the candidate image and any accompanying text as a single "
        "dense vector for matching a text query. The final token is used as "
        "the embedding."
    ),
    ("composed_image_retrieval", "query"): (
        "Encode the reference image and modification text as a single dense "
        "vector for retrieving the modified target image. The final token is "
        "used as the embedding."
    ),
    ("composed_image_retrieval", "target"): (
        "Encode the candidate image as a single dense vector for matching a "
        "reference-image modification query. The final token is used as the "
        "embedding."
    ),
    ("visual_grounding", "query"): (
        "Encode the image and referring expression as a single dense vector "
        "for retrieving the referred region. The final token is used as the "
        "embedding."
    ),
    ("visual_grounding", "target"): (
        "Encode the candidate image region as a single dense vector for "
        "matching a referring expression. The final token is used as the "
        "embedding."
    ),
    ("image_retrieval", "query"): (
        "Encode the multimodal query as a single dense vector for retrieving "
        "the matching image. The final token is used as the embedding."
    ),
    ("image_retrieval", "target"): (
        "Encode the candidate image as a single dense vector for retrieval. "
        "The final token is used as the embedding."
    ),
    ("text_retrieval", "query"): (
        "Encode the query as a single dense vector for retrieving the matching "
        "candidate. The final token is used as the embedding."
    ),
    ("text_retrieval", "target"): (
        "Encode the candidate as a single dense vector for retrieval. The "
        "final token is used as the embedding."
    ),
    ("generic", "query"): (
        "Encode the query as a single dense vector for matching the correct "
        "candidate. The final token is used as the embedding."
    ),
    ("generic", "target"): (
        "Encode the candidate as a single dense vector for matching the query. "
        "The final token is used as the embedding."
    ),
}


# These expressions remove only the legacy task directive.  Captions,
# questions, dialogue, modifications and referring expressions remain as the
# user payload.  Matching is anchored to avoid deleting semantic content.
_QUERY_PREFIXES = {
    "classification": [
        r"Represent the given image for classification\s*[.:]?\s*",
        r"Represent the given news image with\s+(?:the\s+)?following caption "
        r"for domain classification\s*:\s*",
        r"Represent the given image for binary classification to determine "
        r"whether it constitutes hateful speech or not\s*[.:]?\s*",
        r"Identify the object shown in the image\s*[.:]?\s*",
        r"Identify the scene shown in the image\s*[.:]?\s*",
    ],
    "question_answering": [
        r"Represent the given image with the following question\s*:\s*",
    ],
    "image_to_text": [
        r"Find an image caption describing the given (?:everyday|news) image\s*[.:]?\s*",
        r"Find a caption for the news in the given photo\s*[.:]?\s*",
    ],
    "text_to_image": [
        r"Retrieve an image of this news caption\s*[.:]\s*",
        r"Represent the given dialogue about an image, which is used for "
        r"image retrieval\s*:\s*",
        r"Find a Wikipedia image that answers this question\s*:\s*",
        r"Find me (?:an?|the) (?:everyday|news)?\s*image that matches "
        r"(?:the given|this) (?:caption|text)\s*:\s*",
        r"Find me (?:an?|the) (?:everyday|news)?\s*image that "
        r"(?:matches|answers) (?:the given|this) "
        r"(?:news\s+)?(?:caption|text|dialog|question)\s*:\s*",
    ],
    "composed_image_retrieval": [
        r"Given an image, find a similar (?:everyday|fashion)?\s*image with "
        r"the described changes\s*:\s*",
    ],
    "visual_grounding": [
        r"Select the portion of the image that follows the language expressions?\s*[.:]?\s*",
        r"Select the portion of the image that isolates the object labeled as\s*",
        r"Find the image region described by\s*:\s*",
    ],
    "image_retrieval": [
        r"Find a day-to-day image that looks similar to the provided image\s*[.:]?\s*",
    ],
}

_TARGET_PREFIXES = {
    "classification": [r"Represent the class label\s*:\s*"],
    "question_answering": [r"Represent the answer\s*:\s*"],
    "image_to_text": [r"Represent the image caption\s*:\s*"],
    "visual_grounding": [
        r"Represent the given cropped image of the object\s*[.:]?\s*",
    ],
}

_IMAGE_ONLY_TARGET = re.compile(
    r"^Represent the given (?:cropped )?image(?:\s+for[^.]*)?\s*[.]?\s*$",
    flags=re.IGNORECASE,
)


def validate_profile(profile: str) -> str:
    if profile not in {LEGACY_PROFILE, EMBEDDING_V2_PROFILE}:
        raise ValueError(
            f"Unknown Qwen3.5 prompt profile {profile!r}; expected "
            f"{LEGACY_PROFILE!r} or {EMBEDDING_V2_PROFILE!r}"
        )
    return profile


def task_for_dataset(dataset_name: str) -> str:
    return TASK_BY_DATASET.get(_key(dataset_name), "generic")


def instruction_for(dataset_name: str, side: str) -> str:
    if side not in {"query", "target"}:
        raise ValueError(f"Prompt side must be query/target, got {side!r}")
    return PROMPTS[(task_for_dataset(dataset_name), side)]


def clean_legacy_payload(text: Optional[str], dataset_name: str, side: str) -> str:
    """Remove the old task directive while retaining the semantic payload."""
    clean = (text or "").strip()
    task = task_for_dataset(dataset_name)
    patterns = _QUERY_PREFIXES.get(task, []) if side == "query" else _TARGET_PREFIXES.get(task, [])
    for pattern in patterns:
        updated, count = re.subn(
            r"^\s*" + pattern,
            "",
            clean,
            count=1,
            flags=re.IGNORECASE,
        )
        if count:
            return updated.strip()
    if side == "target" and _IMAGE_ONLY_TARGET.fullmatch(clean):
        return ""
    return clean


def prompt_config(profile: str) -> dict:
    validate_profile(profile)
    payload = {
        "implementation": PROMPT_IMPLEMENTATION,
        "profile": profile,
        "pooling_contract": "final_active_token",
        "task_by_dataset": dict(sorted(TASK_BY_DATASET.items())),
        "prompts": {
            f"{task}/{side}": value
            for (task, side), value in sorted(PROMPTS.items())
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_checkpoint_config(output_dir: str, profile: str) -> None:
    if profile == LEGACY_PROFILE:
        return
    path = os.path.join(output_dir, PROMPT_CONFIG_NAME)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(prompt_config(profile), handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def validate_checkpoint_config(checkpoint_dir: str, profile: str) -> None:
    validate_profile(profile)
    path = os.path.join(checkpoint_dir, PROMPT_CONFIG_NAME)
    if profile == LEGACY_PROFILE:
        if os.path.isfile(path):
            raise RuntimeError(
                f"Checkpoint {checkpoint_dir} uses a non-legacy Qwen3.5 prompt "
                "profile; evaluate/resume it with the matching prompt script"
            )
        return
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Qwen3.5 prompt experiment requires checkpoint metadata: {path}"
        )
    with open(path, "r", encoding="utf-8") as handle:
        stored = json.load(handle)
    expected = prompt_config(profile)
    if stored != expected:
        raise RuntimeError(
            "Qwen3.5 prompt profile differs from checkpoint metadata: "
            f"checkpoint={stored} current={expected}"
        )
