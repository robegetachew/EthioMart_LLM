# data_loader.py

def load_and_process_labeled_data(file_path):
    """
    Load and process the labeled data in CoNLL format from a file.

    Args:
    - file_path (str): Path to the labeled file.

    Returns:
    - sentences (list): A list of sentences, each containing a list of (token, label) tuples.
    """
    sentences = []  # List to store sentences
    sentence = []  # List to store tokens in a sentence
    
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        if line == '':  # Blank line means end of sentence
            if sentence:
                sentences.append(sentence)
                sentence = []
        else:
            token, label = line.split()
            sentence.append((token, label))  # Add the token and label tuple to the sentence
    
    # Append the last sentence if the file doesn't end with a blank line
    if sentence:
        sentences.append(sentence)
    
    return sentences


def save_data_as_conll(sentences, output_file):
    """
    Save the structured data in CoNLL format to a file.

    Args:
    - sentences (list): A list of sentences, each containing a list of (token, label) tuples.
    - output_file (str): Path to the output file where data will be saved.
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence in sentences:
            for token, label in sentence:
                f.write(f"{token} {label}\n")
            f.write("\n")  # Blank line between sentences

def process_labeled_data(input_file, output_file):
    """
    Process the labeled data from the input file and save it in CoNLL format.
    
    Args:
    - input_file (str): Path to the labeled data file (e.g., "labeled_telegram_product_price_location.txt").
    - output_file (str): Path to the output file to save the processed data (e.g., "structured_labeled_data.conll").
    """
    # Load and process the labeled data from the file
    structured_data = load_and_process_labeled_data(input_file)
    
    # Save the structured data in CoNLL format
    save_data_as_conll(structured_data, output_file)
    
    print(f"Processed data saved to {output_file}")