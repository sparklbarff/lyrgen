#!/usr/bin/env python3
import argparse
from lyrgen.core import extract_midi_features, generate_lyrics

def main():
    parser = argparse.ArgumentParser(description="Generate lyrics from MIDI + prompt")
    parser.add_argument("--midi", required=True, help="Path to a MIDI file")
    parser.add_argument("--prompt", default="", help="Seed text for generation")
    parser.add_argument("--length", type=int, default=50,
                        help="Number of tokens to generate")
    args = parser.parse_args()

    feats = extract_midi_features(args.midi)
    lyrics = generate_lyrics(feats, args.prompt, max_length=args.length)
    print(lyrics)

if __name__ == "__main__":
    main()
