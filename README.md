# Hallucination Detection System

A Python-based system for detecting and validating AI model responses against a known knowledge base to identify potential hallucinations.

## Project Overview

This project implements a hallucination detection system that validates AI model responses against a predefined knowledge base. The system helps identify when an AI model generates responses that deviate from known facts or ventures into domains where factual validation is not possible.

## Components

- `ask_model.py`: Handles interactions with the AI model
- `validator.py`: Implements the validation logic
- `kb.json`: Contains the knowledge base with factual Q&A pairs
- `model_responses.json`: Stores model responses and validation results
- `run.log`: Detailed logging of the validation process

## Knowledge Base

The system uses a curated knowledge base (`kb.json`) containing 10 factual Q&A pairs covering various topics such as:
- Scientific facts (e.g., chemical formulas, physical constants)
- Historical facts (e.g., authorship, artistic works)
- Geographic information
- Natural phenomena

## Validation Process

The system performs two types of validations:
1. **Knowledge Base Validation**: Compares model responses against known facts
2. **Edge Case Detection**: Identifies responses to questions outside the knowledge domain

### Edge Cases Tested
- Future predictions
- Fictional/mythological topics
- Abstract/philosophical questions
- Scientifically unmeasurable concepts

## Results

Based on the latest validation run:
- Total responses validated: 15
- Hallucinations detected: 14
- Accuracy rate: 6.67%

The system implements a retry mechanism when hallucinations are detected, attempting to get the model to provide more accurate responses.

## Requirements

- Python 3.10+
- OpenAI API key (environment variable)
- Required Python packages (see requirements.txt)

## Usage

1. Set up your OpenAI API key in the environment variables
2. Run the validator:
```bash
python validator.py
```

3. Check the results in `run.log` and `model_responses.json`

## Logging

The system maintains detailed logs (`run.log`) of:
- API requests
- Question-answer pairs
- Validation status
- Retry attempts and responses

## Error Handling

The system includes robust error handling for:
- API failures
- Missing environment variables
- Invalid responses
- Out-of-domain questions

## Future Improvements

- Expand the knowledge base
- Implement more sophisticated validation algorithms
- Add support for multiple AI models
- Enhance edge case detection
- Improve retry strategies

## License

