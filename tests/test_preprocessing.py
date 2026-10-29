from src.preprocessing import clean_text

def test_clean_text_collapses_whitespace():
    assert clean_text("  hello\n  world  ") == "hello world"

def test_clean_text_replaces_url():
    assert "URLTOKEN" in clean_text("Read https://example.com/article now")

def test_clean_text_handles_none():
    assert clean_text(None) == ""
