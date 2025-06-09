"""
Core functions for MIDI/spectral lyric generation.
"""

import pretty_midi
import numpy as np
import librosa
from functools import lru_cache
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def extract_midi_features(midi_path: str) -> dict:
    """
    Load a MIDI file and return key features:
    - tempo (BPM)
    - pitch class histogram
    - average note duration
    """
    try:
        pm = pretty_midi.PrettyMIDI(midi_path)
    except Exception:
        return {
            "tempo": 0.0,
            "pitch_hist": [0] * 128,
            "avg_duration": 0.0,
        }

    tempos = pm.get_tempo_changes()[1]
    tempo = float(np.mean(tempos)) if len(tempos) else 0.0

    pitches = []
    durations = []
    for inst in pm.instruments:
        for note in inst.notes:
            pitches.append(note.pitch)
            durations.append(note.end - note.start)

    features = {
        "tempo": tempo,
        "pitch_hist": list(np.bincount(pitches, minlength=128)),
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
    }

    # Optional spectral analysis if corresponding WAV is available
    try:
        y, sr = librosa.load(midi_path.replace(".mid", ".wav"), sr=None)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        features["mfcc_mean"] = mfccs.mean(axis=1).tolist()
        features["spectral_centroid_mean"] = float(np.mean(centroids))
    except Exception:
        features["mfcc_mean"] = []
        features["spectral_centroid_mean"] = 0.0

    return features


def extract_spectral_features(audio_path: str, n_mfcc: int = 13) -> dict:
    """
    Extract spectral descriptors from an audio file using librosa.
    Returns MFCC means and spectral centroid mean.
    """
    y, sr = librosa.load(audio_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    return {
        "mfcc_mean": mfccs.mean(axis=1).tolist(),
        "spectral_centroid_mean": float(np.mean(centroids)),
    }


@lru_cache(maxsize=1)
def _load_model(model_name: str = "gpt2"):
    """
    Load and cache the tokenizer and model.
    """
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


# Cache the model globally
_TOKENIZER, _MODEL = _load_model()


def generate_lyrics(
    features: dict,
    prompt: str,
    max_length: int = 50,
    top_p: float = 0.9,
    temperature: float = 1.0,
) -> str:
    """
    Generate lyric text seeded by prompt plus extracted features.
    """
    seed = (
        f"Tempo at {features['tempo']:.1f}, "
        f"AvgDur: {features['avg_duration']:.2f}. "
        f"{prompt}"
    )
    # Tokenize with attention mask
    inputs = _TOKENIZER(seed, return_tensors="pt")
    outputs = _MODEL.generate(
        **inputs,
        # determine input length fallback for non-tensor tokens
        max_length=(
            inputs["input_ids"].shape[-1]
            if hasattr(inputs["input_ids"], "shape")
            else len(inputs["input_ids"])
        )
        + max_length,
        do_sample=True,
        top_p=top_p,
        temperature=temperature,
    )
    text = _TOKENIZER.decode(outputs[0], skip_special_tokens=True)
    return text
