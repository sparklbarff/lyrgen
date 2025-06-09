import pytest
from lyrgen.core import extract_midi_features, generate_lyrics


@pytest.fixture
def empty_midi(tmp_path):
    path = tmp_path / "empty.mid"
    path.write_bytes(b"")  # invalid but tests handling
    return str(path)


def test_extract_midi_features_returns_keys(empty_midi):
    feats = extract_midi_features(empty_midi)
    assert isinstance(feats, dict)


def test_generate_lyrics_includes_prompt():
    feats = {
        "tempo": 120,
        "pitch_hist": [0],
        "avg_duration": 0,
    }
    text = generate_lyrics(feats, "Hello")
    assert "Hello" in text
    assert "Tempo at 120.0" in text
