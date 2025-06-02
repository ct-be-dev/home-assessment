#!/usr/bin/env python3
"""
Task 1: Tag sentences based on exact keyword matching (ignoring case)
"""

import csv
import ast
import os

def load_tags(tags_file):
    """
    Load tags and their keywords from CSV file
    """
    tags_dict = {}
    with open(tags_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            tag_id, tag_name, keywords_str = row
            # Convert string representation of list to actual list
            keywords = ast.literal_eval(keywords_str)
            # Convert all keywords to lowercase for case-insensitive matching
            keywords = [keyword.lower() for keyword in keywords]
            tags_dict[tag_name] = keywords
    return tags_dict

def tag_sentence(sentence, tags_dict):
    """
    Tag a sentence with all matching tags based on exact keyword matching
    """
    sentence_lower = sentence.lower()
    matched_tags = []
    
    for tag_name, keywords in tags_dict.items():
        for keyword in keywords:
            if keyword.lower() in sentence_lower:
                matched_tags.append(tag_name)
                break  # Once a tag matches, no need to check other keywords for this tag
    
    return matched_tags

def process_sentences(sentences_file, tags_dict, output_file):
    """
    Process all sentences and write results to output file
    """
    with open(sentences_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        for line in f_in:
            sentence = line.strip()
            if not sentence:
                continue
                
            matched_tags = tag_sentence(sentence, tags_dict)
            
            # Format output as specified: sentence\ttag1, tag2, tag3
            if matched_tags:
                f_out.write(f"{sentence}\t{', '.join(matched_tags)}\n")
            else:
                f_out.write(f"{sentence}\t\n")

def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tags_file = os.path.join(current_dir, "data", "tags.csv")
    sentences_file = os.path.join(current_dir, "data", "sentences.txt")
    output_file = os.path.join(current_dir, "task_1_output.tsv")
    
    # Load tags
    tags_dict = load_tags(tags_file)
    
    # Process sentences
    process_sentences(sentences_file, tags_dict, output_file)
    
    print(f"Task 1 completed. Output saved to {output_file}")

if __name__ == "__main__":
    main()
