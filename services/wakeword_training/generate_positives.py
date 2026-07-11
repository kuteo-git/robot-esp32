"""Generate synthetic "Mai ơi" positive clips.

Must be run with services/.venv/bin/python (has `vieneu` installed):
    services/.venv/bin/python services/wakeword_training/generate_positives.py \
        --out-dir services/wakeword_training/data/positive
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from audio_variants import build_variants
from phrases import POSITIVE_PHRASE
from tts_generate import TTSBackend, generate_dataset, make_vieneu_backend


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[], TTSBackend] = make_vieneu_backend,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/positive")
    args = parser.parse_args(argv)

    backend = backend_factory()
    variants = build_variants()
    written = generate_dataset(
        texts=[POSITIVE_PHRASE],
        variants=variants,
        backend=backend,
        out_dir=Path(args.out_dir),
        label_prefix="pos",
    )
    print(f"Wrote {len(written)} positive clips to {args.out_dir}")


if __name__ == "__main__":
    main()
