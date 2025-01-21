import pandas as pd

class DataCleaner:
    def clean_and_structure(self, data):
        df = pd.DataFrame(data)
        structured_data = df[['Channel Username', 'ID', 'Message', 'Date']]
        return structured_data
