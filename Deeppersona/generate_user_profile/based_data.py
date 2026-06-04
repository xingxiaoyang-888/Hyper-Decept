import json
import os
import random
from typing import Dict, List, Optional, Union, Any
from geonamescache import GeonamesCache
from config import get_completion, parse_gpt_response, parse_json_response, extract_json_from_markdown, parse_nested_json_response

_occupations_cache = None

def get_occupations() -> List[str]:
    """Retrieves occupations from a local JSON file using caching."""
    global _occupations_cache

    if _occupations_cache is not None:
        return _occupations_cache

    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        file_path = os.path.join(project_root, 'data', 'occupations_english.json')
        with open(file_path, 'r', encoding='utf-8') as f:
            _occupations_cache = json.load(f)
        return _occupations_cache
    except Exception as e:
        print(f"Error loading occupations data: {e}")
        return []

def generate_age_info() -> Dict[str, Union[int, str]]:
    """Generates a random age (7-85) and assigns a corresponding age group."""
    age = random.randint(7, 85)

    if age <= 6:
        age_group = "toddler"
    elif age <= 12:
        age_group = "child"
    elif age <= 19:
        age_group = "adolescent"
    elif age <= 29:
        age_group = "young_adult"
    elif age <= 45:
        age_group = "adult"
    elif age <= 65:
        age_group = "middle_aged"
    else:
        age_group = "senior"

    return {
        "age": age,
        "age_group": age_group
    }

def generate_career_info(age: int) -> Dict[str, str]:
    """Generates occupation status. Uses GPT for ages < 18 or > 65, otherwise samples from the local DB."""
    if age < 18 or age > 65:
        prompt = f"Generate an appropriate occupation or status for a {age} year old person. "
        if age < 18:
            prompt += "Consider that the individuals are likely in school or engaged in youth activities. They may not have any formal occupation. If appropriate, you can mention their student status or indicate they have no occupation yet. Only in some cases, consider potential interest in early employment opportunities, internships, or non-traditional educational paths."
        else:
            prompt += "Consider they might be retired but could still be active in various ways."

        messages = [
            {"role": "system", "content": "You are an AI that generates realistic occupation statuses based on age. Respond with just the status, no explanation."},
            {"role": "user", "content": prompt}
        ]

        status = get_completion(messages)
        if not status:
            return {"status": ""}
        return {"status": status}

    occupations = get_occupations()
    if not occupations:
        return {"status": ""}

    career_status = random.choice(occupations)
    return {"status": career_status}

def generate_location() -> Dict[str, str]:
    """Generates a realistic country and city using GeoNames."""
    gc = GeonamesCache()

    countries = gc.get_countries()
    country_code = random.choice(list(countries.keys()))
    country = countries[country_code]

    cities = gc.get_cities()
    country_cities = [city for city in cities.values() if city['countrycode'] == country_code]

    if not country_cities:
        return {
            "country": country['name'],
            "city": "Unknown City"
        }

    city_data = random.choice(country_cities)

    return {
        "country": country['name'],
        "city": city_data['name']
    }

def generate_gender() -> str:
    """Randomly generates gender (male/female)."""
    return random.choice(['male', 'female'])


def generate_personal_values(age: int, gender: str, occupation: str, location: Dict[str, str]) -> Dict[str, str]:
    """Generates core values via GPT based on demographic data."""
    value_type = random.choice(['positive', 'negative', 'neutral'])
    
    prompt = f"""
    Generate a concise description of a person's core values and belief system based on:
    Age: {age}, Gender: {gender}, Occupation: {occupation}, Location: {location['city']}, {location['country']}

    IMPORTANT: This person has a {value_type.upper()} value system. Their values may be entirely consistent with their personal background or may conflict with it. Avoid introducing unnecessary contrasts or contradictions in their beliefs. Try to avoid being related to the community as much as possible. Avoid using words with similar meanings to ‘balance’ and ‘balance’.

    Please generate a short phrase that clearly captures the essence of this person's core values and beliefs without adding conflicting ideas or turnarounds.

    CRITICAL: You must format your response EXACTLY as a valid JSON object with this structure:
    {{
        "values_orientation": "short phrase describing their values"
    }}

    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    
    messages = [
        {"role": "system", "content": "You are an assistant that generates realistic human value systems in ONE SENTENCE, including both positive and negative values. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.8)
        
        result = parse_gpt_response(
            response, 
            expected_fields=["values_orientation"], 
            field_defaults={"values_orientation": ""}
        )
        
        return {
            "values_orientation": result.get("values_orientation") or response.strip()
        }
    except Exception as e:
        print(f"\nError in generate_personal_values: {e}")
        raise

def generate_life_attitude(age: int = None, gender: str = None, occupation: str = None, 
                        location: Dict[str, str] = None, values_orientation: str = None) -> Dict[str, Union[str, Dict, bool]]:
    """Generates life attitude via GPT based on demographics and core values."""
    
    prompt = f"""
    Generate specific attributes about a person's life attitude based on the following information:
    
    Age: {age}
    Gender: {gender}
    Occupation: {occupation}
    Location: {location['city']}, {location['country']}
    Core Values: {values_orientation}
    
    IMPORTANT: This person's attitude toward life can be positive, neutral, or negative. In a negative state, they may hold a pessimistic, cynical, or even nihilistic view of life. Avoid involving concepts such as community or balance. Avoid using words with similar meanings to ‘balance’ and ‘balance’.
    
    I need you to generate ONLY the following specific attributes, each expressed as a single sentence:
    
    1. attitude: A single, concise sentence (5-10 words) describing their overall life attitude
    2. attitude_details: A single sentence (15-20 words) explaining how this attitude manifests in their daily life
    3. coping_mechanism: A single sentence (5-10 words) describing how they deal with challenges
    
    CRITICAL: You must format your response EXACTLY as a valid JSON object with this structure:
    {{"attitude": "single sentence", "attitude_details": "single sentence", "coping_mechanism": "single sentence"}}
    
    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    
    messages = [
        {"role": "system", "content": "You are an assistant that generates realistic human life attitudes in ONE SENTENCE, including positive, neutral, and negative outlooks. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.8)
        
        result = parse_gpt_response(
            response,
            expected_fields=["attitude", "attitude_details", "coping_mechanism"],
            field_defaults={
                "attitude": "",
                "attitude_details": "",
                "coping_mechanism": ""
            }
        )
        
        for field in ["attitude", "attitude_details", "coping_mechanism"]:
            if not result[field]:
                raise ValueError(f"Missing required field: {field}")
        
        attitude = result["attitude"]
        attitude_details = result["attitude_details"]
        coping_mechanism = result["coping_mechanism"]
    except Exception as e:
        print(f"\nError in generate_life_attitude: {e}")
        raise
    
    return {
        "attitude": attitude,
        "attitude_details": attitude_details,
        "coping_mechanism": coping_mechanism
    }

def generate_personal_story(age: int, gender: str, occupation: str, location: Dict[str, str], 
                                    values_orientation: str, life_attitude: Dict[str, str]) -> Dict[str, str]:
    """Generates a personal story via GPT based on demographic data and life attitude."""
    attitude = life_attitude.get("attitude", "")
    attitude_category = life_attitude.get("attitude_category", "neutral")
    
    num_stories = random.randint(1, 3)
    
    prompt = f"""
    Generate {num_stories} concise personal stories for a person with the following characteristics:
    
    Age: {age}
    Gender: {gender}
    Occupation: {occupation}
    Location: {location['city']}, {location['country']}
    Core Values: {values_orientation}
    Life Attitude: {attitude} ({attitude_category})
    
    IMPORTANT: The story can be positive, negative, or a mix of both. Please do not avoid including life experiences that may be controversial or have negative consequences. The narration should be as specific as possible, objective, and free from any subjective comments or value judgments. The stories that unfold should be closely related to their country and region, reflecting events that could genuinely happen to the people there. This could be a random event unrelated to the background, or a significant turning point in their life. Please avoid including anything related to community building.
    
    Please provide {num_stories} brief personal stories (each 150-200 words).
    
    CRITICAL: You must format your response EXACTLY as a valid JSON object with this structure:
    {{"personal_stories": ["story 1", "story 2", ...]}}
    
    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    messages = [
        {"role": "system", "content": "You are an assistant that generates concise realistic personal stories, including both positive and negative life experiences. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.8)
        
        result = parse_gpt_response(
            response,
            expected_fields=["personal_stories"],
            field_defaults={
                "personal_stories": []
            }
        )
        
        stories = result["personal_stories"]
        
        if not stories:
            raise ValueError("Failed to generate personal stories")
        
        combined_story = "\n\n".join([f"Story {i+1}: {story}" for i, story in enumerate(stories)])
        
        return {
            "personal_story": combined_story
        }
    except Exception as e:
        print(f"\nError in generate_personal_story: {e}")
        raise

def generate_interests_and_hobbies(personal_story: Dict[str, Any]) -> Dict[str, Any]:
    """Infers hobbies or interests based on the generated personal story."""
    if not personal_story or not isinstance(personal_story, dict):
        raise ValueError("Personal story data must be provided to generate hobbies")
    
    story_text = personal_story.get("personal_story", "")
    
    prompt = f"""
   Based on the following personal story and key life events, infer two to three hobbies or interests this person might use to relax. These activities can be positive or negative and may include non-traditional, controversial, or unexpected ones, such as various sports, traveling, or even smoking, drinking, or using marijuana. Please make inferences about the person’s possible interests based on the story, rather than simply extracting them directly from the story. 
   
     Personal Story: {story_text}

     IMPORTANT: Avoid including anything related to community-building activities.
     
    Please extract 2 hobbies or interests based on these reflections and format your response as a JSON object:

     {{
        "interests": ["interest1", "interest2"]
     }}

    DO NOT include any text before or after the JSON. The response must be parseable by json.loads().
    """
    
    messages = [
        {"role": "system", "content": "You are an assistant that extracts realistic interests and hobbies from a person's life story, including both positive activities and negative habits. You ALWAYS respond with valid JSON objects that can be parsed by json.loads()."}, 
        {"role": "user", "content": prompt}
    ]
    
    try:
        response = get_completion(messages, temperature=0.2)
        
        from config import parse_gpt_response
        result = parse_gpt_response(
            response, 
            expected_fields=["interests"], 
            field_defaults={"interests": []}
        )
        
        interests = result.get("interests", [])
    except Exception as e:
        print(f"\nError in generate_interests_and_hobbies: {e}")
        raise
    
    print(f"\nInterests generated: {len(interests)}")
    
    return {
        "interests": interests
    }