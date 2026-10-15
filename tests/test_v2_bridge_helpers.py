from scripts.run_vimax_v2_action_bridge import (
    _build_canonicalization_metadata,
    _duration_source_label,
    _normalize_duration_context,
)


def test_duration_context_is_sorted_and_rejects_nonpositive_values() -> None:
    normalized = _normalize_duration_context(
        {
            1: {
                "shot_idx": 1,
                "duration_seconds": "4.5",
                "duration_status": "director_estimated",
            },
            0: {
                "shot_idx": 0,
                "duration_seconds": 0,
                "duration_status": "not_provided",
            },
        }
    )

    assert [item["shot_idx"] for item in normalized] == [0, 1]
    assert normalized[0]["duration_seconds"] is None
    assert normalized[0]["duration_status"] == "not_provided"
    assert normalized[1]["duration_seconds"] == 4.5


def test_duration_labels_do_not_upgrade_estimates_to_verified_timing() -> None:
    assert "not verified" in _duration_source_label("director_estimated")
    assert "placeholder" in _duration_source_label("not_provided")


def test_structural_pass_does_not_imply_production_ready() -> None:
    metadata = _build_canonicalization_metadata([], [])

    assert metadata["structural_status"] == "PASS"
    assert metadata["text_continuity_status"] == "NOT VERIFIED"
    assert metadata["production_ready"] is False
