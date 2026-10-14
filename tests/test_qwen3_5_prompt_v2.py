import json

import pytest

from src.training.qwen3_5_prompt import (
    EMBEDDING_V2_PROFILE,
    PROMPTS,
    clean_legacy_payload,
    instruction_for,
    prompt_config,
    task_for_dataset,
    validate_checkpoint_config,
    write_checkpoint_config,
)


def test_all_36_mmeb_datasets_have_an_explicit_task():
    datasets = [
        "ImageNet-1K", "N24News", "HatefulMemes", "SUN397", "VOC2007",
        "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
        "Wiki-SS-NQ", "VisDial", "CIRR", "VisualNews_t2i",
        "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA",
        "OVEN", "FashionIQ", "EDIS", "OK-VQA", "A-OKVQA", "DocVQA",
        "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "GQA",
        "TextVQA", "VizWiz", "MSCOCO", "RefCOCO", "RefCOCO-Matching",
        "Visual7W-Pointing",
    ]
    assert len(datasets) == 36
    assert all(task_for_dataset(name) != "generic" for name in datasets)


def test_training_dataset_alias_and_roles_resolve_consistently():
    assert task_for_dataset("mmeb/ImageNet_1K") == "classification"
    assert instruction_for("mmeb/ImageNet_1K", "query") == PROMPTS[
        ("classification", "query")
    ]
    assert instruction_for("ImageNet-1K", "target") == PROMPTS[
        ("classification", "target")
    ]


@pytest.mark.parametrize(
    "dataset,side,text,expected",
    [
        (
            "ImageNet_1K",
            "query",
            "Represent the given image for classification",
            "",
        ),
        (
            "OK-VQA",
            "query",
            "Represent the given image with the following question: What color is the car?",
            "What color is the car?",
        ),
        ("OK-VQA", "target", "Represent the answer: red", "red"),
        (
            "MSCOCO_i2t",
            "target",
            "Represent the image caption: a dog on a beach",
            "a dog on a beach",
        ),
        (
            "CIRR",
            "query",
            "Given an image, find a similar everyday image with the described changes: make it blue",
            "make it blue",
        ),
        ("MSCOCO_t2i", "target", "Represent the given image.", ""),
        (
            "RefCOCO",
            "query",
            "Select the portion of the image that follows the language expression: man in red",
            "man in red",
        ),
        (
            "N24News",
            "query",
            "Represent the given news image with following caption for domain "
            "classification: an election-night photograph",
            "an election-night photograph",
        ),
        (
            "HatefulMemes",
            "query",
            "Represent the given image for binary classification to determine "
            "whether it constitutes hateful speech or not",
            "",
        ),
        ("VOC2007", "query", "Identify the object shown in the image", ""),
        ("SUN397", "query", "Identify the scene shown in the image", ""),
        (
            "VisDial",
            "query",
            "Represent the given dialogue about an image, which is used for "
            "image retrieval: Q:is this a child A:yes",
            "Q:is this a child A:yes",
        ),
        (
            "VisualNews_t2i",
            "query",
            "Retrieve an image of this news caption. A minister arrives.",
            "A minister arrives.",
        ),
        (
            "VisualNews_i2t",
            "query",
            "Find a caption for the news in the given photo.",
            "",
        ),
        (
            "WebQA",
            "query",
            "Find a Wikipedia image that answers this question: Which tower is taller?",
            "Which tower is taller?",
        ),
        (
            "NIGHTS",
            "query",
            "Find a day-to-day image that looks similar to the provided image.",
            "",
        ),
        (
            "MSCOCO",
            "query",
            'Select the portion of the image that isolates the object labeled as "zebra"',
            '"zebra"',
        ),
        (
            "MSCOCO",
            "target",
            "Represent the given cropped image of the object",
            "",
        ),
    ],
)
def test_legacy_directive_is_removed_but_semantic_payload_is_preserved(
    dataset, side, text, expected
):
    assert clean_legacy_payload(text, dataset, side) == expected


def test_unknown_text_is_never_destructively_rewritten():
    text = "A rare user-authored sentence that must survive."
    assert clean_legacy_payload(text, "unknown-dataset", "query") == text


def test_checkpoint_config_roundtrip_and_mismatch_detection(tmp_path):
    write_checkpoint_config(str(tmp_path), EMBEDDING_V2_PROFILE)
    validate_checkpoint_config(str(tmp_path), EMBEDDING_V2_PROFILE)
    path = tmp_path / "qwen3_5_prompt_config.json"
    stored = json.loads(path.read_text())
    assert stored == prompt_config(EMBEDDING_V2_PROFILE)

    stored["pooling_contract"] = "wrong"
    path.write_text(json.dumps(stored))
    with pytest.raises(RuntimeError):
        validate_checkpoint_config(str(tmp_path), EMBEDDING_V2_PROFILE)
