#!/usr/bin/env python3

import argparse
import os
import sys
import logging

from audio_to_midi import converter


def _convert_beat_to_time(bpm, beat):
    try:
        parts = beat.split("/")
        if len(parts) > 2:
            raise Exception()

        beat = [int(part) for part in parts]
        fraction = beat[0] / beat[1]
        bps = bpm / 60
        ms_per_beat = bps * 1000
        return fraction * ms_per_beat
    except Exception:
        raise RuntimeError("Invalid beat format: {}".format(beat))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="The sound file to process.")
    parser.add_argument(
        "--output", "-o", help="The MIDI file to output. Default: <infile>.mid"
    )
    parser.add_argument(
        "--bpm", "-b", type=int, help="Beats per minute. Defaults: 60", default=60
    )
    parser.add_argument(
        "--beat",
        "-B",
        help="Time window in terms of beats (1/4, 1/8, etc.). Supercedes the time window parameter.",
    )
    parser.add_argument(
        "--no-progress", "-n", action="store_true", help="Don't print the progress bar."
    )
    args = parser.parse_args()

    args.output = (
        "{}.mid".format(os.path.basename(args.infile))
        if not args.output
        else args.output
    )

    if args.single_note:
        args.note_count = 1

    if args.pitch_set:
        for key in args.pitch_set:
            if key not in range(12):
                raise RuntimeError("Key values must be in the range: [0, 12)")

    if args.beat:
        args.time_window = _convert_beat_to_time(args.bpm, args.beat)
        print(args.time_window)

    if args.pitch_range:
        if args.pitch_range[0] > args.pitch_range[1]:
            raise RuntimeError("Invalid pitch range: {}".format(args.pitch_range))

    if args.condense_max:
        args.condense = True

    return args


def main(infile):
    try:
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")

        #args = parse_args()

        process = converter.Converter(
            infile=infile,
            outfile = infile+".mid",
            progress=None
        )
        process.convert()
#     except KeyboardInterrupt:
#         sys.exit(1)
    except Exception as e:
        logging.exception(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
