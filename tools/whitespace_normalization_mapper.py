VARIOUS_WHITESPACES = {
    ' ', '	', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ',
    ' ', ' ', ' ', '　', '​', '‌', '‍', '⁠', '￼', ''
}



def normalize_whitespace(text):
    """
    Normalize different kinds of whitespaces to whitespace ' ' (0x20) in a given text string.

    :param text: The input text string to be processed.
    :return: The processed text string with normalized whitespaces.
    """
    # remove whitespaces before and after the main content
    stripped_text = text.strip()

    # replace all kinds of whitespaces with ' '
    normalized_text = ''.join([' ' if char in VARIOUS_WHITESPACES else char for char in stripped_text])

    return normalized_text


