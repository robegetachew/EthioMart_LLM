import re

class TextPreprocessor:
    @staticmethod
    def preprocess_text(text):
        # Remove unwanted characters and normalize
        text = re.sub(r'[^\u1200-\u137F\s]', '', text)  # Keep only Amharic characters
        text = text.lower()  # Normalize to lowercase
        return text.split()  # Tokenize into words

    def preprocess_dataframe(self, df, column_name='Message'):
        df['Processed Message'] = df[column_name].apply(self.preprocess_text)
        return df
