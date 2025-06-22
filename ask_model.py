from openai import OpenAI
import json
from typing import Dict, List, Tuple
import logging
from dotenv import load_dotenv, dotenv_values

# Load environment variables
config = dotenv_values(".env")

# Configure logging
logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class ModelInteractor:
    def __init__(self, api_key: str = None):
        """Initialize the ModelInteractor with OpenAI client."""
        # Get API key from .env file
        self.api_key = api_key or config.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found in .env file")
            
        self.client = OpenAI(api_key=self.api_key)
        self.load_knowledge_base()
        
    def load_knowledge_base(self) -> None:
        """Load the knowledge base from kb.json."""
        try:
            with open('kb.json', 'r') as f:
                self.kb = json.load(f)
        except FileNotFoundError:
            raise Exception("kb.json not found. Please ensure the knowledge base file exists.")

    def get_model_response(self, question: str) -> str:
        """Get response from the model for a given question."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",  # Using GPT-4 for better accuracy
                messages=[
                    {"role": "system", "content": "You are a helpful assistant. Answer questions directly and concisely."},
                    {"role": "user", "content": question}
                ],
                temperature=0.1  # Low temperature for more consistent answers
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error getting model response: {str(e)}")
            return None

    def process_questions(self) -> List[Dict]:
        """Process all questions including KB questions and edge cases."""
        results = []
        
        # Process KB questions
        for qa_pair in self.kb['qa_pairs']:
            question = qa_pair['question']
            response = self.get_model_response(question)
            results.append({
                'question': question,
                'model_response': response,
                'type': 'kb_question',
                'kb_answer': qa_pair['answer']
            })
            logging.info(f"KB Question: {question}")
            logging.info(f"Model Response: {response}")
            logging.info(f"KB Answer: {qa_pair['answer']}")
            logging.info("-" * 50)

        # Edge case questions
        edge_cases = [
            "What is the capital of the ancient Atlantis?",
            "How many atoms are in a human thought?",
            "What color is the number 7?",
            "What is the sound of one hand clapping?",
            "Can you explain quantum physics to a goldfish?"
        ]

        # Process edge cases
        for question in edge_cases:
            response = self.get_model_response(question)
            results.append({
                'question': question,
                'model_response': response,
                'type': 'edge_case',
                'kb_answer': None
            })
            logging.info(f"Edge Case Question: {question}")
            logging.info(f"Model Response: {response}")
            logging.info("-" * 50)

        return results

def main():
    """Main function to run the model interaction."""
    try:
        interactor = ModelInteractor()
        results = interactor.process_questions()
        
        # Save results for validator
        with open('responses.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print("Processing complete. Check run.log for details.")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 