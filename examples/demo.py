import argparse
from lyrgen.core import extract_midi_features, generate_lyrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi", required=True)
    parser.add_argument("--prompt", default="")
    args = parser.parse_args()

    feats = extract_midi_features(args.midi)
    print(generate_lyrics(feats, args.prompt))

if __name__ == "__main__":
    main()
