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
    feats = {"tempo": 120.0, "pitch_hist": [0], "avg_duration": 0.0}
    text = generate_lyrics(feats, "Hello")
    assert "Hello" in text
    assert "Tempo at 120.0" in text


# ────────────────────────────────────────────────────────────────────────────────
# Smoke‐test for generate_lyrics model integration
# ────────────────────────────────────────────────────────────────────────────────
class DummyTokenizer:
    def __call__(self, text, return_tensors):
        # return a fake PyTorch‐style dict
        return {
            "input_ids": text,  # we’ll treat the seed string itself as “ids”
            "attention_mask": text,  # same for mask
        }

    def decode(self, seq, skip_special_tokens):
        # echo back what was passed
        return seq


class DummyModel:
    def generate(self, *args, **kwargs):
        # we only care that **kwargs got through
        return [kwargs["input_ids"]]


def test_generate_lyrics_model_runs(monkeypatch):
    """
    Ensure generate_lyrics can call through the HF interface
    without actually loading GPT-2.
    """
    import lyrgen.core as core

    # swap out the real tokenizer + model
    monkeypatch.setattr(core, "_TOKENIZER", DummyTokenizer())
    monkeypatch.setattr(core, "_MODEL", DummyModel())

    feats = {"tempo": 90.0, "pitch_hist": [0], "avg_duration": 0.25}
    out = generate_lyrics(feats, "FooBar", max_length=5)

    # DummyModel returned our seed string
    assert isinstance(out, str)
    assert out == "Tempo at 90.0, AvgDur: 0.25. FooBar"
