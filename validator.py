from openai import OpenAI
import json
import logging
from typing import Dict, List, Optional
import difflib
from dotenv import load_dotenv, dotenv_values

# Load environment variables
config = dotenv_values(".env")

# Configure logging
logging.basicConfig(
    filename='run.log',
    level=logging.INFO,
    format='%(asctime)s - %(message)s'
)

class HallucinationValidator:
    def __init__(self, api_key: str = None):
        """Initialize the validator with OpenAI client for retries."""
        # Get API key from .env file
        self.api_key = api_key or config.get("OPENAI_API_KEY")
        
        if not self.api_key:
            raise ValueError("OpenAI API key not found in .env file")
            
        self.client = OpenAI(api_key=self.api_key)
        self.load_knowledge_base()
        self.load_responses()
        
    def load_knowledge_base(self) -> None:
        """Load the knowledge base from kb.json."""
        with open('kb.json', 'r') as f:
            self.kb = json.load(f)
            
    def load_responses(self) -> None:
        """Load the model responses from responses.json."""
        with open('responses.json', 'r') as f:
            self.responses = json.load(f)

    def string_similarity(self, str1: str, str2: str) -> float:
        """Calculate string similarity using difflib."""
        return difflib.SequenceMatcher(None, 
            str1.lower(), 
            str2.lower()
        ).ratio()

    def validate_response(self, question: str, model_response: str, kb_answer: Optional[str]) -> Dict:
        """Validate a single response against KB or mark as out-of-domain."""
        if kb_answer:  # Question is from KB
            similarity = self.string_similarity(model_response, kb_answer)
            
            if similarity < 0.8:  # Threshold for considering it different
                # Retry with more specific prompt
                retry_response = self.retry_question(question, kb_answer)
                return {
                    'status': 'RETRY: answer differs from KB',
                    'similarity': similarity,
                    'retry_response': retry_response
                }
            return {
                'status': 'VALID',
                'similarity': similarity,
                'retry_response': None
            }
        else:  # Edge case question
            return {
                'status': 'RETRY: out-of-domain',
                'similarity': None,
                'retry_response': self.retry_question(question)
            }

    def retry_question(self, question: str, kb_answer: Optional[str] = None) -> str:
        """Retry the question with more specific prompt."""
        try:
            if kb_answer:
                prompt = f"Please answer this question accurately. The correct answer should be similar to: {kb_answer}\nQuestion: {question}"
            else:
                prompt = f"This question may be out of domain. If you don't have factual information to answer it, please explicitly say so.\nQuestion: {question}"
                
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a careful assistant that prioritizes accuracy over speculation."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error in retry: {str(e)}")
            return "Error during retry"

    def validate_all_responses(self) -> List[Dict]:
        """Validate all responses and log results."""
        validation_results = []
        
        for response in self.responses:
            result = self.validate_response(
                response['question'],
                response['model_response'],
                response.get('kb_answer')
            )
            
            validation_entry = {
                'question': response['question'],
                'original_response': response['model_response'],
                'validation_result': result['status'],
                'retry_response': result['retry_response']
            }
            
            validation_results.append(validation_entry)
            
            # Log the validation
            logging.info(f"Question: {response['question']}")
            logging.info(f"Original Response: {response['model_response']}")
            logging.info(f"Validation Status: {result['status']}")
            if result['retry_response']:
                logging.info(f"Retry Response: {result['retry_response']}")
            logging.info("-" * 50)
        
        return validation_results

def main():
    """Main function to run the validation."""
    try:
        validator = HallucinationValidator()
        results = validator.validate_all_responses()
        
        # Save validation results
        with open('validation_results.json', 'w') as f:
            json.dump(results, f, indent=4)
            
        print("Validation complete. Check run.log for details.")
        
        # Generate summary statistics
        total = len(results)
        hallucinations = sum(1 for r in results if 'RETRY' in r['validation_result'])
        print(f"\nSummary:")
        print(f"Total responses validated: {total}")
        print(f"Hallucinations detected: {hallucinations}")
        print(f"Accuracy rate: {((total-hallucinations)/total)*100:.2f}%")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 