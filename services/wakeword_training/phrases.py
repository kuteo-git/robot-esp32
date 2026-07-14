POSITIVE_PHRASE = "Na Bi ơi"

# Phonetically close to "Na Bi ơi" (+ the previous wake words) — trained as
# explicit negatives so the model fires ONLY on the full "Na Bi ơi", not on either
# half alone or near-miss syllables.
HARD_NEGATIVE_PHRASES = [
    "Na ơi",         # first syllable alone
    "Bi ơi",         # second syllable alone
    "Na Bi",         # no "ơi"
    "Ba Bi ơi",      # near-miss first syllable
    "Ma Bi ơi",
    "La Bi ơi",
    "Na Vi ơi",      # near-miss second syllable
    "Na Mi ơi",
    "Na Ni ơi",
    "Nam Bi ơi",
    "Mai ơi",        # previous wake words -> must NOT fire
    "An An ơi",
]

# Everyday Vietnamese household speech, used as negatives so the model doesn't
# false-trigger during normal conversation.
GENERIC_NEGATIVE_SENTENCES = [
    "Hôm nay trời đẹp quá",
    "Con ăn cơm chưa",
    "Mẹ ơi con đói bụng",
    "Bao giờ mình đi chơi",
    "Anh đang làm gì đấy",
    "Chị lấy giúp em cái này với",
    "Bố đi làm về chưa",
    "Em muốn xem phim hoạt hình",
    "Tối nay ăn gì hả mẹ",
    "Con làm bài tập xong chưa",
]
