"""Core functions for MIDI/spectral lyric generation."""

import pretty_midi
import numpy as np

def extract_midi_features(midi_path: str) -> dict:
    """
    Load a MIDI file and return key features:
    - tempo (BPM)
    - pitch classes histogram
    - note durations
    """
    pm = pretty_midi.PrettyMIDI(midi_path)
    tempo = pm.get_tempo_changes()[1].mean()
    pitches = []
    durations = []
    for inst in pm.instruments:
        for note in inst.notes:
            pitches.append(note.pitch)
            durations.append(note.end - note.start)
    return {
        "tempo": float(tempo),
        "pitch_hist": list(np.bincount(pitches, minlength=128)),
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
    }

def generate_lyrics(features: dict, prompt: str) -> str:
    """
    Given features and an optional prompt,
    returns placeholder lyrics.
    """
    lines = [
        f"Tempo at {features['tempo']:.1f} BPM",
        f"Average duration {features['avg_duration']:.2f}s",
        f"Prompt: {prompt or 'no prompt'}",
    ]
    return "\n".join(lines)
