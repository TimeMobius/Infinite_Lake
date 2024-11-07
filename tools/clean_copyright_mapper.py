import regex as re

class CleanCopyrightMapper:
    """Mapper to clean copyright comments at the beginning of the text samples."""

    def __init__(self):
        """Initialization method."""
        self.pat = re.compile(r'/\*[^*]*\*+(?:[^*/][^*]*\*+)*/')
        self.cpat = re.compile(r'copyright | 版权', re.IGNORECASE)

    def process_text(self, text):
        """
        Process the input text by cleaning copyright comments.

        :param text: Input text string
        :return: Processed text with copyright comments cleaned
        """
        r = self.pat.search(text)
        if r:
            # found one, now see if it contains "copyright", if so strip it
            span = r.span()
            sub = text[span[0]:span[1]]
            if self.cpat.search(sub):
                # cut it
                text = text[:span[0]] + text[span[1]:]

        lines = text.split('\n')
        skip = 0

        # Greedy replace any file that begins with a comment block
        for k in range(len(lines)):
            if (lines[k].startswith('//') or lines[k].startswith('#')
                    or lines[k].startswith('--') or not lines[k]):
                skip += 1
            else:
                break

        if skip:
            # we skipped, consume it
            text = '\n'.join(lines[skip:])
        return text

# Example usage:
#mapper = CleanCopyrightMapper()
#input_text = """
#/* 版权声明 */小明
#"""
#output_text = mapper.process_text(input_text)
#print(output_text)
