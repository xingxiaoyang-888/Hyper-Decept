import json
import time
import os
import re
from typing import Dict
from openai import OpenAI

OPENAI_API_KEY = "OPENAI_API_KEY"

GPT_MODEL = "gpt-4o"

class PersonalizedAttributeExtractor:
    def __init__(self):
        """Initialize OpenAI client and load template categories."""
        self.client = OpenAI(
            api_key=OPENAI_API_KEY,
        )
        
        with open('/home/zhou/persona/dataset/3.20/template.json', 'r') as f:
            template_data = json.load(f)
            self.persona_categories = template_data['persona_categories']

    def extract_attributes(self, question: str, reason: str) -> Dict:
        """Use GPT to analyze questions and reasons to extract required personalization attributes."""
        prompt = f"""
# Task Description
1) For each category of user information, reason about the attributes about the user that could be useful for personalizing this question.
   - List the more attributes the better.
   - Do NOT be too specific for user attributes. High-level and general attributes are preferred.
2) Summarize the useful user attributes as a list of category.attribute_name.sub_attribute_name pairs.
3) Conclude if the given user question could be personalized or not. 

# Requirements
Output each attribute as a 3-level structure (e.g. X.Y.Z)

IMPORTANT: The first level (before the first dot) of each attribute MUST be chosen from the most suitable category in the provided top-level categories list above. No other top-level categories are allowed.

Available Top-Level Categories:
{', '.join(self.persona_categories)}

Question:
{question}

Personalization Reason:
{reason}

Output Format:
{{
    "attributes": [
        "attributeA1.levelA2.levelA3",
        "attributeB1.levelB2.levelB3",
        "..."
    ]
}}

Only return the JSON object in the above format."""

        completion = self.client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert specialized in analyzing and extracting user personalization attributes. You need to carefully analyze questions and reasons to extract relevant personalization attributes."},
                {"role": "user", "content": prompt}
            ]
        )
        
        response_text = completion.choices[0].message.content.strip()
        print(f"\nRaw API response:\n{response_text}")
        
        if '```json' in response_text:
            json_match = re.search(r'```json\s*(.+?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
            else:
                if response_text.startswith('```json'):
                    response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
        elif '```' in response_text:
            json_match = re.search(r'```\s*(.+?)\s*```', response_text, re.DOTALL)
            if json_match:
                response_text = json_match.group(1).strip()
        
        response_text = response_text.strip()
        print(f"\nProcessed response text:\n{response_text}")
        
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            json_obj_match = re.search(r'\{\s*"attributes"\s*:\s*\[.+?\]\s*\}', response_text, re.DOTALL)
            if json_obj_match:
                try:
                    result = json.loads(json_obj_match.group(0))
                    print("Successfully extracted JSON object using regex")
                except json.JSONDecodeError:
                    result = {"attributes": []}
                    print("Failed to parse JSON even after extraction, using empty result")
            else:
                result = {"attributes": []}
                print("Could not find valid JSON object, using empty result")
        
        processed_attributes = []
        for attr in result.get('attributes', []):
            first_level = attr.split('.')[0] if '.' in attr else attr
            
            first_level_variants = [
                first_level,
                first_level.replace('_', ' '),
                first_level.replace('.', ' ')
            ]
            
            matched_category = None
            for variant in first_level_variants:
                if variant in self.persona_categories:
                    matched_category = variant
                    break
            
            if matched_category:
                if '.' in attr:
                    rest_of_attr = '.'.join(attr.split('.')[1:])
                    processed_attributes.append(f"{matched_category}.{rest_of_attr}")
                else:
                    processed_attributes.append(matched_category)
            else:
                processed_attributes.append(attr)
                print(f"Note: Keeping attribute '{attr}' even though first level '{first_level}' is not in template categories")
        
        result['attributes'] = processed_attributes
        
        return result

    def process_questions(self, input_file: str, output_attributes_file: str, output_full_file: str, num_questions: int = 5):
        """Processes questions in a JSON file and extracts personalization attributes."""
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        all_attributes = set()
        processed_data = []
        
        for i, item in enumerate(data[:num_questions]):
            question = item['question']
            reason = item.get('tags', {}).get('is_personalizable', {}).get('reason', '')
            
            result = self.extract_attributes(question, reason)
            print("\nExtracted Attributes:")
            for attr in sorted(result['attributes']):
                print(f"  - {attr}")
            all_attributes.update(result['attributes'])
            
            full_item = {
                'question_id': item.get('question_id'),
                'original_id_in_source': item.get('original_id_in_source'),
                'source': item.get('source'),
                'question': item['question'],
                'original_answer': item.get('original_answer', ''),
                'tags': item.get('tags', {}),
                'personalized_attributes': {
                    'attributes': result['attributes']
                }
            }
            processed_data.append(full_item)
            
            with open(output_full_file, 'w', encoding='utf-8') as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=4)
            
            time.sleep(0.1)
        
        sorted_attributes = sorted(list(all_attributes))
        with open(output_attributes_file, 'w', encoding='utf-8') as f:
            json.dump(sorted_attributes, f, ensure_ascii=False, indent=4)

def main():
    print('Using proxy:', os.environ.get('https_proxy'))
    
    extractor = PersonalizedAttributeExtractor()
    input_file = "PATH.json"
    output_attributes_file = "PATH.json"
    output_full_file = "PATH.json"
    
    extractor.process_questions(input_file, output_attributes_file, output_full_file, num_questions=1224)

if __name__ == "__main__":
    main()