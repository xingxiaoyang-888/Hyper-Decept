import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from build_study_cases import RAW_IDENTIFIER, sanitize_text  # noqa: E402


def test_text_sanitizer_removes_handles_urls_and_long_numbers() -> None:
    result = sanitize_text(
        "Contact @named_account at https://example.org/item/123 or 123456789."
    )

    assert "@named_account" not in result
    assert "https://" not in result
    assert "123456789" not in result
    assert not RAW_IDENTIFIER.search(result)
