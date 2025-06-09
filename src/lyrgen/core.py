import pretty_midi
import numpy as np

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
        # on empty/invalid file, return zeroed defaults
        return {
            "tempo": 0.0,
            "pitch_hist": [0] * 128,
            "avg_duration": 0.0,
        }

    # successful path:
    tempos = pm.get_tempo_changes()[1]
    tempo = float(np.mean(tempos)) if len(tempos) else 0.0

    pitches = []
    durations = []
    for inst in pm.instruments:
        for note in inst.notes:
            pitches.append(note.pitch)
            durations.append(note.end - note.start)

    return {
        "tempo": tempo,
        "pitch_hist": list(np.bincount(pitches, minlength=128)),
        "avg_duration": float(np.mean(durations)) if durations else 0.0,
    }
