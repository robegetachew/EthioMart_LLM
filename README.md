#EthioMart Data Extraction & Labeling
This project focuses on extracting and labeling data from Ethiopian-based Telegram e-commerce channels for Named Entity Recognition (NER) tasks. The goal is to identify and label entities such as products, prices, and locations from Amharic text in messages.

#Project Overview
The process is divided into the following key steps:
1. Data Collection: Scraping data from Ethiopian Telegram e-commerce channels using a custom-built scraper.
2. Data Preprocessing: Cleaning and formatting data, filtering for Amharic text, tokenizing, and normalizing the messages.

3. Entity Labeling: Manually labeling a subset of messages to identify and categorize entities (products, prices, locations) using the CoNLL format.
4. Output: Saving the labeled data in the CoNLL format for future use in machine learning models.
#Prerequisites
Python 3.7 or higher

#Required libraries:
telethon (for Telegram scraping)
pandas (for data manipulation)
re (for regular expressions)
dotenv (for loading environment variables)
asyncio (for handling asynchronous operations)

You can install the required libraries using:
pip install -r requirements.txt

Setup
Create a .env file: Ensure you have a .env file with your Telegram API credentials. You can obtain these by creating a bot via BotFather.
TG_API_ID=your_api_id
TG_API_HASH=your_api_hash
PHONE=your_phone_number

Run the Telegram Scraper: The scraper collects data from selected Telegram channels. To run it, execute the following script:
python scrape_telegram_data.py

#Data Preprocessing: Preprocess the scraped data by filtering for Amharic messages and tokenizing them. This step cleans and prepares the data for entity labeling.

#Labeling Entities: Manually label the entities in a subset of the dataset following the CoNLL format. Entities include:

B-PRODUCT: Beginning of a product entity
I-PRODUCT: Inside a product entity
B-LOC: Beginning of a location entity
I-LOC: Inside a location entity
B-PRICE: Beginning of a price entity
I-PRICE: Inside a price entity
O: Other tokens that don't belong to any entity
The labeled data is stored in a plain text file (labeled_telegram_product_price_location.txt).

#Files
scrape_telegram_data.py: Script to scrape data from Telegram channels.
preprocess_data.py: Preprocesses the raw data (filters for Amharic text, tokenizes, and normalizes).
labeled_telegram_product_price_location.txt: A text file containing labeled entities in the CoNLL format.
requirements.txt: List of required Python libraries.
