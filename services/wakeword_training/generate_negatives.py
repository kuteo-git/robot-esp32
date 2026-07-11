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

# Negatives use a deliberately smaller non-voice grid than positives. Positives need
# the full prosody grid because "Mai ơi" is the exact phrase the model must recognize
# across many renditions; negatives only need enough acoustic diversity to avoid
# overfitting to a narrow negative set — sampling weights at training time rebalance
# pos/neg importance regardless of raw clip count. Using the full grid here would
# produce 25,200 hard + 36,000 generic clips (~20h of TTS at ~1.2s/clip); this reduced
# grid keeps full voice diversity (6 voices, still important for negatives) while
# cutting the other 4 axes to a representative subset spanning low/baseline/high.
NEGATIVE_TEMPERATURES = [0.7, 1.0, 1.3]
NEGATIVE_TOP_KS = [20, 50]
NEGATIVE_PITCH_SEMITONES = [-4, 0, 4]
NEGATIVE_SPEED_FACTORS = [0.85, 1.15]


def main(
    argv: list[str] | None = None,
    backend_factory: Callable[[], TTSBackend] = make_vieneu_backend,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="data/negative_vi")
    args = parser.parse_args(argv)

    backend = backend_factory()
    variants = build_variants(
        temperatures=NEGATIVE_TEMPERATURES,
        top_ks=NEGATIVE_TOP_KS,
        pitch_semitones=NEGATIVE_PITCH_SEMITONES,
        speed_factors=NEGATIVE_SPEED_FACTORS,
    )
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
