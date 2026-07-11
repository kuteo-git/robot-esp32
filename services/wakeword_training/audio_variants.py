from dataclasses import dataclass

# Confirmed against vieneu/assets/voices.json's "presets" keys (ASCII, no diacritics).
PRESET_VOICES = ["Binh", "Doan", "Ly", "Ngoc", "Tuyen", "Vinh"]

# Grid sized so build_variants() (all defaults) lands in the 3,000-4,500 target
# range for a single phrase: 6 voices * 8 temperatures * 3 top_ks * 5 pitches
# * 5 speeds = 3,600 variants. Temperature gets the most steps since it's the
# axis with the most perceptual payoff for prosodic variety (pacing, pitch
# contour, hesitation) in TTS sampling; top_k, pitch, and speed are widened
# more modestly to cover timbre/register and playback-rate variation without
# blowing up the grid.
TEMPERATURES = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
TOP_KS = [20, 35, 50]
PITCH_SEMITONES = [-4, -2, 0, 2, 4]
SPEED_FACTORS = [0.85, 0.925, 1.0, 1.075, 1.15]


@dataclass(frozen=True)
class GenerationVariant:
    voice: str
    temperature: float
    top_k: int
    pitch_semitones: int
    speed_factor: float

    @property
    def tag(self) -> str:
        return (
            f"{self.voice}_t{self.temperature}_k{self.top_k}"
            f"_p{self.pitch_semitones}_s{self.speed_factor}"
        )


def build_variants(
    voices: list[str] = PRESET_VOICES,
    temperatures: list[float] = TEMPERATURES,
    top_ks: list[int] = TOP_KS,
    pitch_semitones: list[int] = PITCH_SEMITONES,
    speed_factors: list[float] = SPEED_FACTORS,
) -> list["GenerationVariant"]:
    return [
        GenerationVariant(voice, temperature, top_k, pitch, speed)
        for voice in voices
        for temperature in temperatures
        for top_k in top_ks
        for pitch in pitch_semitones
        for speed in speed_factors
    ]
