#!/usr/bin/env python3
"""
Task 2: Tag sentences using machine learning approach
"""

import csv
import ast
import os
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Filter out FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def load_tags(tags_file):
    """
    Load tags and their keywords from CSV file
    """
    tags_dict = {}
    tag_names = []
    with open(tags_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            tag_id, tag_name, keywords_str = row
            # Convert string representation of list to actual list
            keywords = ast.literal_eval(keywords_str)
            tags_dict[tag_name] = keywords
            tag_names.append(tag_name)
    return tags_dict, tag_names

def load_sentences(sentences_file):
    """
    Load sentences from file
    """
    sentences = []
    with open(sentences_file, 'r', encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            if sentence:
                sentences.append(sentence)
    return sentences

def create_training_data(sentences, tags_dict, tag_names):
    """
    Create training data by using keyword matching as initial labels
    and augmenting with examples from keywords themselves
    """
    X = []  # Sentences
    y = []  # Labels (list of tags for each sentence)
    
    # First, add all sentences with their keyword-based tags
    for sentence in sentences:
        sentence_lower = sentence.lower()
        matched_tags = []
        
        for tag_name, keywords in tags_dict.items():
            for keyword in keywords:
                if keyword.lower() in sentence_lower:
                    matched_tags.append(tag_name)
                    break
        
        X.append(sentence)
        y.append(matched_tags)
    
    # Then, augment training data with keywords as examples
    for tag_name, keywords in tags_dict.items():
        for keyword in keywords:
            # Add all keywords as examples (both single words and phrases)
            X.append(keyword)
            y.append([tag_name])
            
            # Add some variations with common prefixes/suffixes for robustness
            if len(keyword.split()) > 1:  # Only for phrases
                X.append(f"I need help with {keyword}")
                y.append([tag_name])
                X.append(f"Information about {keyword}")
                y.append([tag_name])
    
    # Make sure all tags have at least some positive examples
    tag_counts = {tag: 0 for tag in tag_names}
    for tags in y:
        for tag in tags:
            tag_counts[tag] += 1
    
    # For any tag with too few examples, add more synthetic examples
    for tag, count in tag_counts.items():
        if count < 5:  # Ensure at least 5 examples per tag
            keywords = tags_dict[tag]
            for i in range(5 - count):
                if i < len(keywords):
                    X.append(f"I have a question about {keywords[i]}")
                    y.append([tag])
    
    return X, y

def train_model(X, y, tag_names):
    """
    Train a multi-label classifier using TF-IDF features
    """
    # Convert multi-label format to binary format
    mlb = MultiLabelBinarizer(classes=tag_names)
    y_bin = mlb.fit_transform(y)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y_bin, test_size=0.2, random_state=42)
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
        max_features=10000,
        min_df=1,
        max_df=0.9
    )
    
    # Transform text to TF-IDF features
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    
    # Train multi-label classifier with LogisticRegression (more stable than SVC for this case)
    classifier = OneVsRestClassifier(LogisticRegression(
        C=5.0,
        solver='liblinear',
        random_state=42,
        max_iter=1000
    ))
    classifier.fit(X_train_tfidf, y_train)
    
    # Evaluate on validation set
    y_pred = classifier.predict(X_val_tfidf)
    print("Model Validation Report:")
    print(classification_report(y_val, y_pred, target_names=tag_names, zero_division=0))
    
    return vectorizer, classifier, mlb

def predict_tags(sentences, vectorizer, classifier, mlb):
    """
    Predict tags for sentences using the trained model
    """
    X_tfidf = vectorizer.transform(sentences)
    y_pred_bin = classifier.predict(X_tfidf)
    y_pred = mlb.inverse_transform(y_pred_bin)
    return y_pred

def save_output(sentences, predicted_tags, output_file):
    """
    Save tagged sentences to output file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for sentence, tags in zip(sentences, predicted_tags):
            if tags:
                f.write(f"{sentence}\t{', '.join(tags)}\n")
            else:
                f.write(f"{sentence}\t\n")

def main():
    # File paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tags_file = os.path.join(current_dir, "data", "tags.csv")
    sentences_file = os.path.join(current_dir, "data", "sentences.txt")
    output_file = os.path.join(current_dir, "task_2_output.tsv")
    model_dir = os.path.join(current_dir, "models")
    
    # Create models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load tags and sentences
    tags_dict, tag_names = load_tags(tags_file)
    sentences = load_sentences(sentences_file)
    
    # Create training data
    X, y = create_training_data(sentences, tags_dict, tag_names)
    
    # Train model
    vectorizer, classifier, mlb = train_model(X, y, tag_names)
    
    # Save models
    joblib.dump(vectorizer, os.path.join(model_dir, "vectorizer.pkl"))
    joblib.dump(classifier, os.path.join(model_dir, "classifier.pkl"))
    joblib.dump(mlb, os.path.join(model_dir, "mlb.pkl"))
    
    # Predict tags for all sentences
    predicted_tags = predict_tags(sentences, vectorizer, classifier, mlb)
    
    # Save output
    save_output(sentences, predicted_tags, output_file)
    
    print(f"Task 2 completed. Output saved to {output_file}")
    print(f"Models saved to {model_dir}")

if __name__ == "__main__":
    main()
