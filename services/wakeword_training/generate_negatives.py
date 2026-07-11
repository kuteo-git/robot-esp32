"""Generate synthetic Vietnamese negative clips (hard near-misses + generic speech).

Must be run with services/.venv/bin/python (has `vieneu` installed):
    services/.venv/bin/python services/wakeword_training/generate_negatives.py \
        --out-dir services/wakeword_training/data/negative_vi
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from audio_variants import build_variants
from phrases import GENERIC_NEGATIVE_SENTENCES, HARD_NEGATIVE_PHRASES
from tts_generate import TTSBackend, generate_dataset, make_vieneu_backend


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[], TTSBackend] = make_vieneu_backend,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/negative_vi")
    args = parser.parse_args(argv)

    backend = backend_factory()
    variants = build_variants()
    out_dir = Path(args.out_dir)

    hard = generate_dataset(
        texts=HARD_NEGATIVE_PHRASES,
        variants=variants,
        backend=backend,
        out_dir=out_dir / "hard",
        label_prefix="hardneg",
    )
    generic = generate_dataset(
        texts=GENERIC_NEGATIVE_SENTENCES,
        variants=variants,
        backend=backend,
        out_dir=out_dir / "generic",
        label_prefix="genneg",
    )
    print(f"Wrote {len(hard)} hard-negative clips and {len(generic)} generic-negative clips to {args.out_dir}")


if __name__ == "__main__":
    main()
