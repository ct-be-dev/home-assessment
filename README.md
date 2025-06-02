# Solution to Home Assessment

This solution implements both tasks as requested in the assignment:

## Implementation Details

### Task 1: Exact Keyword Matching
The implementation in `task1.py` performs the following steps:
1. Loads the tags and their keywords from `data/tags.csv`
2. Processes each sentence from `data/sentences.txt`
3. For each sentence, finds all matching tags based on exact keyword matching (ignoring case)
4. Outputs the results to `task_1_output.tsv` in the required format

### Task 2: Machine Learning Approach
The implementation in `task2.py` uses a machine learning approach:
1. Uses TF-IDF vectorization to convert sentences into feature vectors
2. Implements a multi-label classification approach using LinearSVC
3. Trains the model using the keyword matches as initial labels
4. Augments the training data with the keywords themselves as examples
5. Outputs the results to `task_2_output.tsv` in the required format

## How to Run

### Prerequisites
Make sure you have the required dependencies installed:
```
pip install -r requirements.txt
```

### Running Task 1
```
python task1.py
```
This will generate the `task_1_output.tsv` file with the tagged sentences.

### Running Task 2
```
python task2.py
```
This will:
1. Train a machine learning model
2. Save the model components in the `models` directory
3. Generate the `task_2_output.tsv` file with the tagged sentences

## Notes
- The machine learning approach in Task 2 uses the keyword matches from Task 1 as initial training data
- It further augments the training data with multi-word keywords to improve classification
- The model uses TF-IDF features with unigrams and bigrams to capture phrase patterns
- A multi-label classifier is used since sentences can have multiple tags
