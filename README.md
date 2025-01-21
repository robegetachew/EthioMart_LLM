# EthioMart-Telegram-based-e-commerce-
## Objectives
The primary objective of this project is to set up a data ingestion system that fetches messages from Ethiopian-based Telegram e-commerce channels and prepares the raw data for entity extraction. Additionally, we will label a subset of the data using the CoNLL format for Named Entity Recognition (NER).

## Project Overview
This project for interim submission is divided into two main tasks:

1. **Data Ingestion and Preprocessing:**
   1. Identify and connect to relevant Telegram e-commerce channels using a custom scraper.
   2. Implement a message ingestion system to collect text, images, and documents posted in real-time.
   3. Preprocess the text data by tokenizing, normalizing, and handling Amharic-specific linguistic features.
   4. Clean and structure the data into a unified format, separating metadata (e.g., sender, timestamp) from message content.
   5. Store the preprocessed data in a structured format for further analysis.

2. **Labeling Dataset in CoNLL Format:**
   - Identify and label entities such as products, price, and location in Amharic text using the CoNLL format.
   - The labeling should include:
     - B-Product: Beginning of a product entity.
     - I-Product: Inside a product entity.
     - B-LOC: Beginning of a location entity.
     - I-LOC: Inside a location entity.
     - B-PRICE: Beginning of a price entity.
     - I-PRICE: Inside a price entity.
     - O: Tokens outside any entities.
   - Save the labeled dataset in a plain text file using the CoNLL format.

## Selected Channels
The following Ethiopian-based Telegram e-commerce channels have been selected for data ingestion:
- **@Fashiontera**

**Project Folder Structure as template**
```|   .gitignore
|   ProjectFolderStr.txt
|   README.md
|   requirements.txt
|   
+---.github
|   \---workflows
+---.vscode
|       settings
|       
+---notebooks
+---scripts
|       __init__.py
|       
+---src
|       __init__.py
|       
\---tests
        __init__.py
```

## Tools and Libraries
- **Python**: The primary programming language used for the implementation.
- **Telethon**: A Python library for interacting with Telegram’s API to scrape messages.
- **Pandas**: For data manipulation and storage in structured formats.
- **NLTK or SpaCy**: For text preprocessing and tokenization specific to Amharic linguistic features.

## Conclusion
This project aims to create a systematic approach to extract useful data from Ethiopian-based Telegram channels, focusing on e-commerce. By automating the data ingest process and facilitating the labeling of relevant entities, we aim to enhance the training data for potential machine learning models in the area of Natural Language Processing (NLP) dealing with Amharic text.

### Insights
- The need for accurate data extraction from regional platforms like Telegram is crucial for effective business and model training.
- Adjustments may be necessary during the scraping and preprocessing steps to accommodate specific linguistic characteristics of the Ethiopian context.

## Future Work
- Fine-tuning the scraper for better efficiency and accuracy.
- Model Comparison & Selection
- Model Interpretability
- Exploring advanced NLP techniques to improve entity recognition.
- Gathering additional data for training and evaluation of machine learning models.

## License


## Contact
For inquiries, please contact [jenberligab@gmail.com](mailto:jenberligab@gmail.com).
