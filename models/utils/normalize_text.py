from string import punctuation, whitespace


SPACE_SYMBOL: str = " "


def normalize_text(text: str) -> str:
    for sym in punctuation:
        text = text.replace(sym, SPACE_SYMBOL)

    for sym in whitespace:
        text = text.replace(sym, SPACE_SYMBOL)

    while SPACE_SYMBOL * 2 in text:
        text = text.replace(SPACE_SYMBOL * 2, SPACE_SYMBOL)

    text = text.lower().strip()

    text = text.replace("ё", "е") # Игнорировать различие в записи Е и Ё

    return text