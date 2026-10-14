import json

from judge_mmeb_pairs_workflow import (
    Case,
    build_workflow_payload,
    build_cases,
    clean_text,
    parse_judge_output,
    summarize,
)


def test_build_cases_uses_candidate_zero_and_aligned_prediction():
    rows = [
        {
            "qry_text": "<|image_1|>\nquestion",
            "qry_img_path": "GQA/q.jpg",
            "tgt_text": ["gold", "other"],
            "tgt_img_path": ["", ""],
        }
    ]
    cases = build_cases("GQA", rows, [("alternative", "")])
    assert len(cases) == 1
    assert cases[0].positive_text == "gold"
    assert cases[0].predict_text == "alternative"
    assert cases[0].exact_top1_match is False
    assert clean_text("<|image_1|>\nhello") == "hello"


def test_parse_judge_output_accepts_workflow_json():
    output = json.dumps(
        {
            "positive": {
                "judgment": "true_positive",
                "confidence": 0.9,
                "reason": "valid",
            },
            "predict": {
                "category": "hard_negative",
                "confidence": 0.8,
                "reason": "wrong",
            },
        }
    )
    parsed, error = parse_judge_output(output)
    assert error == ""
    assert parsed["positive"]["judgment"] == "true_positive"
    assert parsed["predict"]["judgment"] == "hard_negative"


def test_workflow_payload_uses_ordered_multi_image_protocol(tmp_path):
    case = Case(
        case_id="EDIS:row-0",
        dataset="EDIS",
        split="OOD",
        task_type="text_to_image_retrieval",
        row_index=0,
        query_text="caption",
        query_image_path="",
        positive_text="gold image",
        positive_image_path="",
        predict_text="predicted image",
        predict_image_path="",
        exact_top1_match=False,
    )
    payload = build_workflow_payload(case, tmp_path, "agent", 1280, 88, 2_500_000)
    assert payload["case_id"] == case.case_id
    assert payload["image_base64_list"] == []
    query = json.loads(payload["query"])
    assert query["image_roles"] == []
    assert query["candidate_positive_text"] == "gold image"
    assert set(payload) == {
        "agentCode", "stream", "case_id", "image_base64_list", "query"
    }
    assert "query_image" not in payload


def test_workflow_payload_deduplicates_exact_candidate_image(tmp_path):
    from PIL import Image

    image_path = tmp_path / "candidate.jpg"
    Image.new("RGB", (8, 8), "red").save(image_path)
    case = Case(
        case_id="EDIS:row-1",
        dataset="EDIS",
        split="OOD",
        task_type="text_to_image_retrieval",
        row_index=1,
        query_text="caption",
        query_image_path="",
        positive_text="gold image",
        positive_image_path="candidate.jpg",
        predict_text="gold image",
        predict_image_path="candidate.jpg",
        exact_top1_match=True,
    )
    payload = build_workflow_payload(case, tmp_path, "agent", 1280, 88, 2_500_000)
    query = json.loads(payload["query"])
    assert query["image_roles"] == ["positive+predict"]
    assert len(payload["image_base64_list"]) == 1
    assert payload["image_base64_list"][0].startswith("data:image/jpeg;base64,")


def test_summary_counts_semantic_false_negative_and_bad_gold():
    rows = [
        {
            "qry_text": "q",
            "qry_img_path": "GQA/q.jpg",
            "tgt_text": ["gold"],
            "tgt_img_path": [""],
        }
    ]
    case = build_cases("GQA", rows, [("alternative", "")])[0]
    results = {
        case.case_id: {
            "status": "ok",
            "positive_judgment": "false_positive",
            "predict_category": "false_negative",
        }
    }
    summary = summarize([case], results)
    assert summary["overall"]["invalid_designated_positive"] == 1
    assert summary["overall"]["semantic_false_negative"] == 1
