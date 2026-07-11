POSITIVE_PHRASE = "Mai ơi"

# Phonetically close to "Mai ơi" — trained as explicit negatives so the model
# doesn't false-trigger on near-miss Vietnamese speech.
HARD_NEGATIVE_PHRASES = [
    "Mài ơi",
    "Hai ơi",
    "Mai ới",
    "mai ơi anh ơi",
    "Nai ơi",
    "Mai đi",
    "Bài ơi",
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
