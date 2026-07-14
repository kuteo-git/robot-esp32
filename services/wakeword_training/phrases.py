POSITIVE_PHRASE = "An An ơi"

# Phonetically close to "An An ơi" (and the previous wake word) — trained as
# explicit negatives so the model only fires on the DOUBLED "An An ơi", not on
# single "an", other doubled names, or near-miss household speech.
HARD_NEGATIVE_PHRASES = [
    "An ơi",         # single (not doubled) — the key discriminator
    "An An",         # doubled but no "ơi"
    "ăn cơm chưa",   # "ăn" collides with "an"
    "bình an",
    "an toàn",
    "an ninh",
    "Ba ba ơi",      # a different doubled name
    "Hân Hân ơi",    # near-doubled name
    "Lan ơi",
    "Nam ơi",
    "Mai ơi",        # the previous wake word -> must NOT fire
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
