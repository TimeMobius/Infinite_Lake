import re


def split_sentence(text):
    text = re.sub('([.。！!？\?])([^’”])', r'\1\n\2', text)  # noqa
    text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)  # noqa
    text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)  # noqa
    text = re.sub('([.。!！？\?\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)  # noqa
    return text.split('\n')



def split_sentence(text):
    # Split the text into sentences using various sentence delimiters
    text = re.sub('([.。！!?])([^’”])', r'\1\n\2', text)
    text = re.sub('(\.{6})([^’”])', r'\1\n\2', text)
    text = re.sub('(\…{2})([^’”])', r'\1\n\2', text)
    text = re.sub('([.。!！??\.{6}\…{2}][’”])([^’”])', r'\1\n\2', text)
    return text.split('\n')

def remove_repeat_sentences(text, lowercase=False, ignore_special_character=True, min_repeat_sentence_length=3):
    """
    Removes repeat sentences in the given text.

    :param text: The input text to process.
    :param lowercase: Whether to convert the text to lower case.
    :param ignore_special_character: Whether to ignore special characters when judging repeated sentences.
    :param min_repeat_sentence_length: Sentences shorter than this length will not be deduplicated.
    :return: The processed text with repeat sentences removed.
    """
    # Split the text into lines
    lines = text.split('\n')
    new_lines = []
    hash_set = set()
    for line in lines:
        new_sent = ''
        if line:
            # Split the line into sentences
            sentences = split_sentence(line)
            for sentence in sentences:
                copy = sentence.strip()
                if lowercase:
                    copy = copy.lower()
                if ignore_special_character:
                    copy = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '', copy)

                if len(copy) < min_repeat_sentence_length:
                    new_sent += sentence
                elif copy not in hash_set:
                    new_sent += sentence
                    hash_set.add(copy)
        new_lines.append(new_sent)

    # Join the lines back into a single string
    processed_text = '\n'.join(new_lines)
    return processed_text


