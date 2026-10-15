import re

import pytest


_QUOTED_LINE_RE = re.compile(r"\u201c([^\u201d]+)\u201d")


def _spoken_order(text: str) -> list[str]:
    return _QUOTED_LINE_RE.findall(text)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known V8 candidate defect: motion_desc reverses dialogue order and "
        "ViMax currently has no automatic motion/audio order validator."
    ),
)
def test_v8_dialogue_order_regression_is_reproduced_with_synthetic_fixture() -> None:
    audio_desc = (
        "[\u8bf4\u8bdd\u4eba] Role A: \u201cFirst line.\u201d\n"
        "[\u8bf4\u8bdd\u4eba] Role B: \u201cSecond line.\u201d"
    )
    v8_style_motion_desc = (
        "Role B says \u201cSecond line.\u201d "
        "Role A then says \u201cFirst line.\u201d"
    )

    assert _spoken_order(v8_style_motion_desc) == _spoken_order(audio_desc)
