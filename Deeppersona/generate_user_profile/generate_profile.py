#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import random
import sys
import time
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional
from config import get_completion
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def safe_str(value):
    """
    Ensures a string is returned.
      - If value is already a str, return it.
      - Otherwise (dict, list, or other), return the JSON serialization with multi-line formatting.
    """
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2)

def get_project_root() -> str:
    """Gets the absolute path of the project root directory."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))
    return project_root

def copy_files_from_source_to_target():
    """Sets up and verifies the target output directory."""
    correct_output_dir = os.path.join(get_project_root(), "output")
    os.makedirs(correct_output_dir, exist_ok=True)
    print(f"Output directory set to: {correct_output_dir}")
    return True

def get_timestamped_filename(base_path: str) -> str:
    """Appends a timestamp to a given file path.
    
    Args:
        base_path: The base file path.
        
    Returns:
        str: The file path with a timestamp appended.
    """
    directory = os.path.dirname(base_path)
    filename = os.path.basename(base_path)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"{name}_{timestamp}{ext}"
    return os.path.join(directory, timestamped_filename)

def save_json_file(file_path: str, data: Dict, use_timestamp: bool = True) -> str:
    """Saves data to a JSON file.
    
    Args:
        file_path: The target file path.
        data: The dictionary data to save.
        use_timestamp: Whether to append a timestamp to the filename.
        
    Returns:
        str: The actual file path used to save the data.
    """
    try:
        actual_path = get_timestamped_filename(file_path) if use_timestamp else file_path
        os.makedirs(os.path.dirname(actual_path), exist_ok=True)
        with open(actual_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return actual_path
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        return file_path

def extract_paths(obj: Dict, prefix: str = "") -> List[str]:
    """Extracts all attribute paths from a nested JSON object.
    
    Args:
        obj: The nested JSON object.
        prefix: The current path prefix.
        
    Returns:
        List[str]: A list of attribute paths.
    """
    paths = []
    for key, value in obj.items():
        new_prefix = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if not value: 
                paths.append(new_prefix)
            else:
                paths.extend(extract_paths(value, new_prefix))
    return paths

def generate_category_attributes(category_paths: Dict, custom_prompt: str, category_name: str) -> Dict:
    """Generates all attribute values for a top-level category in a single prompt.
    
    Args:
        category_paths: The structure of attribute paths under the top-level category.
        custom_prompt: The custom prompt containing specific generation instructions.
        category_name: The name of the top-level category.
        
    Returns:
        Dict: A dictionary of all generated attribute values.
    """
    leaf_paths = []
    
    def collect_leaf_paths(obj, current_path):
        for key, value in obj.items():
            path = f"{current_path}.{key}" if current_path else key
            if isinstance(value, dict):
                if not value:
                    leaf_paths.append(path)
                else:
                    collect_leaf_paths(value, path)
    
    collect_leaf_paths(category_paths, "")
    
    if not leaf_paths:
        return {}
    
    system_prompt = "Format your response as a JSON object where each key is the attribute path and each value is the generated attribute value (not exceeding 100 characters)."
    
    user_prompt = f"{custom_prompt}\n\nAttribute Paths to generate values for:\n"
    for path in leaf_paths:
        user_prompt += f"- {path}\n"
    user_prompt += "\nGenerate suitable values for all these attributes in JSON format."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    try:
        print(f"Generating {len(leaf_paths)} attributes for {category_name}...")
        response = get_completion(messages)
        if not response:
            print(f"Failed to generate attributes for {category_name}: Empty response.")
            return {}
            
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith("```json"):
                cleaned_response = cleaned_response[7:]
            if cleaned_response.endswith("```"):
                cleaned_response = cleaned_response[:-3]
            cleaned_response = cleaned_response.strip()
            
            generated_values = json.loads(cleaned_response)
            print(f"Successfully generated {len(generated_values)} attributes.")
            return generated_values
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON for {category_name}: {e}")
            print(f"Response snippet: {response[:100]}..." if len(response) > 100 else f"Response: {response}")
            return {}
    except Exception as e:
        print(f"Error generating attributes for {category_name}: {e}")
        return {}

def generate_final_summary(profile: Dict, base_info: Dict = None) -> str:
    """Generates the final summary for the user profile.
    
    Args:
        profile: The complete user profile data.
        base_info: Basic information, including life story elements.
        
    Returns:
        str: The final summary text.
    """
    system_prompt = """
Your task: Based solely on the provided user attributes and personal story, create an objective and factual personal profile, strictly between 150–400 words.

Content Requirements:
    •   The profile must be written entirely in the first-person perspective.
    •   The output should be a coherent, logically structured narrative, not a list of points. The order may vary: it does not need to follow the fixed “background → challenge → conclusion” pattern, and may instead begin with daily life or interests.
    •   The opening must explicitly state my country or region, ensuring that geographic location is clearly highlighted at the very start.
    •   Must include:
    1.  Basic background (e.g., location, identity)
    2.  Daily life or work routines
    3.  Personal interests and hobbies (explicitly highlighted)
    4.  Behavioral tendencies or values (positive or negative)
    •   Interests and hobbies must be integrated naturally, not superficially. Add small, ordinary details (e.g., food preferences, leisure activities, quirks) that make the character feel real.
    •   If there are negative traits, imperfections, or contradictions, they must be represented faithfully without softening. Do not reframe them as “growth” or “lessons learned.”
    •   No declarative or reflective endings. Avoid abstract statements like “I’ve learned…,” “This shows…,” or “Success means….” The ending should remain grounded in daily routines or interests.
    •   Only include information explicitly provided in the attributes and story. No invention, speculation, or interpretation.
    •   Prohibit the use of words such as 'balance' and 'balance'
"""
    user_prompt = f"Complete Profile (in JSON format):\n{json.dumps(profile, ensure_ascii=False, indent=2)}\n\n"
    
    user_prompt +="""Generate a first-person narrative of 100-400 words from the provided profile. Your primary goal is to make the person feel real, believable, and authentic.

To achieve this, strictly follow the 'Show, Don't Tell' principle:
1.  **Illustrate, Don't Declare:** Show values and traits through specific actions, stories, and decisions, rather than stating them directly.
2.  **Connect Actions to Motivation:** Briefly explain the 'why' behind key life choices and habits to reveal the person's inner logic and create narrative depth.
3.  **Maintain a Natural Voice:** The tone must be sincere and grounded—thoughtful but not overly abstract or dramatic.

Weave all elements into a cohesive story, not a simple list of facts."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = get_completion(messages)
        summary = response.strip() if response else ""
        word_count = len(summary.split())
        
        if word_count < 100:
            print(f"Warning: Summary is only {word_count} words (minimum 100).")
        elif word_count > 400:
            summary = enforce_word_limit(summary, 400)
            print(f"Summary was adjusted to 400 words (from {word_count}).")
        else:
            print(f"Summary generated with {word_count} words.")
        return summary
    except Exception as e:
        print(f"Error generating final summary: {e}")
        return ""

def print_section(section: Dict, indent: int = 0) -> None:
    """Prints the contents of a configuration section.
    
    Args:
        section: The configuration section to print.
        indent: The indentation level.
    """
    indent_str = "  " * indent
    for key, value in section.items():
        if isinstance(value, dict):
            print(f"{indent_str}{key}:")
            print_section(value, indent + 1)
        else:
            print(f"{indent_str}{key}: {value}")

def generate_section(template_section: Dict, base_info: str, section_name: str, indent: int = 0) -> Dict:
    """Generates a section of the configuration profile.
    
    Args:
        template_section: The corresponding section in the template.
        base_info: Basic information text.
        section_name: Name of the section.
        indent: Indentation level.
        
    Returns:
        Dict: The generated configuration section.
    """
    section_result = {}
    indent_str = "  " * indent
    
    print(f"{indent_str}Generating {section_name}...")
    
    if indent == 0:
        all_attributes = generate_category_attributes(template_section, base_info, section_name)
        
        if all_attributes:
            for path, value in all_attributes.items():
                parts = path.split('.')
                if len(parts) > 1 and parts[0] == section_name:
                    parts = parts[1:]
                
                current = section_result
                for i, part in enumerate(parts):
                    if i == len(parts) - 1:
                        current[part] = value
                        print(f"{indent_str}  - {'.'.join(parts)}: {value}")
                    else:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
            
            return section_result
    
    for key, value in template_section.items():
        current_path = f"{section_name}.{key}" if section_name else key
        
        if isinstance(value, dict):
            if not value: 
                generated_value = generate_attribute_value(current_path, base_info)
                section_result[key] = generated_value
                print(f"{indent_str}  - {key}: {generated_value}")
            else: 
                section_result[key] = generate_section(value, base_info, current_path, indent + 1)
    
    return section_result

def enforce_word_limit(text: str, limit: int = 300) -> str:
    """Trims the text to a maximum of `limit` words."""
    words = text.split()
    if len(words) > limit:
        return ' '.join(words[:limit])
    return text

def append_profile_to_json(file_path: str, profile: Dict, use_timestamp: bool = True) -> str:
    """Appends a profile to a JSON file.
    
    Args:
        file_path: The target file path.
        profile: The profile to append.
        use_timestamp: Whether to use a timestamp in the filename.
        
    Returns:
        str: The actual file path used.
    """
    try:
        if use_timestamp:
            actual_path = get_timestamped_filename(file_path)
            profiles = [profile]
        else:
            actual_path = file_path
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
            else:
                profiles = []
            profiles.append(profile)
        
        os.makedirs(os.path.dirname(actual_path), exist_ok=True)
        with open(actual_path, 'w', encoding='utf-8') as f:
            json.dump(profiles, f, ensure_ascii=False, indent=2)
        return actual_path
    except Exception as e:
        print(f"Error appending profile to JSON: {e}")
        return file_path

def generate_single_profile(
    template: Dict = None,
    profile_index: int = 0,
    attribute_count: int = 200,
    output_dir: str = None,
) -> Dict:
    """Generates a complete user profile based on a given template and count.
    
    Args:
        template: Optional template to guide generation.
        profile_index: The index of the profile being generated.
        attribute_count: The desired number of attributes.
        
    Returns:
        Dict: The generated user profile.
    """
    
    print(f'Running select_attributes.py to update base files with {attribute_count} attributes...')
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from select_attributes import generate_user_profile as gen_profile
        from select_attributes import get_selected_attributes, save_results
        
        user_profile = gen_profile()
        selected_paths = get_selected_attributes(user_profile, attribute_count)
        correct_output_dir = output_dir or os.path.join(
            get_project_root(), "output"
        )
        save_results(user_profile, selected_paths, correct_output_dir)
    except Exception as e:
        print(f"Error executing select_attributes functions: {e}")
        return {}

    output_dir = output_dir or os.path.join(get_project_root(), "output")
    base_info_path = os.path.join(output_dir, 'user_profile.json')
    
    with open(base_info_path, 'r', encoding='utf-8') as f:
        base_info = json.load(f)
        
    if 'Occupations' not in base_info:
        print("Warning: 'Occupations' key is missing in the user profile. Setting it to an empty list.")
        base_info['Occupations'] = []

    selected_paths_path = os.path.join(output_dir, 'selected_paths.json')
    with open(selected_paths_path, 'r', encoding='utf-8') as f:
        selected_paths = json.load(f)

    for k in ("life_attitude", "interests"):
        base_info[k] = safe_str(base_info.get(k, ""))

    assert 'Occupations' in base_info, "The 'Occupations' key is missing in the user profile."
    
    profile = {
        "Base Info": base_info,
        "Generated At": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Profile Index": profile_index + 1
    }
    
    life_story = base_info.get("personal_story", {}).get("personal_story", "")
    demographic_input = (
        "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Instructions: Based on the `base_info` and `life_story` provided, **develop and elaborate on** the 'Demographic Information' section in English. Your task is to **appropriately expand upon and enrich** the existing information from `base_info` and incorporate relevant insights from the `life_story`. Focus on elaborating on the given data points, adding further relevant details, or providing context to make the demographic profile more comprehensive and insightful. While you should avoid simply repeating the `base_info` verbatim, ensure that all generated content is **directly built upon and logically extends** the information available in `base_info` and `life_story`, rather than introducing entirely new, unrelated demographic facts. The goal is a coherent, more descriptive, and enhanced version of the original data that reflects the person's life experiences."
    )
    demographic_template = selected_paths.get("Demographic Information")
    if demographic_template and demographic_template != "":
        print('Generating Demographic Information...')
        demographic_section = generate_category_attributes(demographic_template, demographic_input, "Demographic Information")
        nested_result = {}
        for path, value in demographic_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Demographic Information":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Demographic Information"] = nested_result
    else:
        print('No valid "Demographic Information" template found in selected_paths, skipping.')
    
    career_template = selected_paths.get("Career and Work Identity")
    if career_template and career_template != "":
        print('Generating Career and Work Identity...')
        career_input = (
            "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
            "Life Story (for reference):\n" + str(life_story) + "\n\n"
            "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
            "Instructions: Based on the `base_info`, `life_story`, and `Demographic Information` provided above, **develop and elaborate on** the 'Career and Work Identity' section in English. "
            "Your aim is to distill and articulate the career identity, professional journey, and work-related aspirations that are **evident or can be reasonably inferred from the combined `base_info`, `life_story`, and `Demographic Information`**. "
            "Offer fresh insights by providing a **deeper, more nuanced interpretation or by highlighting connections within the provided data** that illuminate these aspects. "
            "Ensure that this elaboration is **logically consistent with and directly stems from** the provided information. "
            "**Do not introduce new career details or aspirations that are not grounded in or clearly supported by the source material.** "
            "The section should be an insightful and coherent expansion of what can be understood from the source material."
        )
        career_info_section = generate_category_attributes(career_template, career_input, "Career and Work Identity")
        nested_result = {}
        for path, value in career_info_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Career and Work Identity":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Career and Work Identity"] = nested_result
    else:
        print('No valid "Career and Work Identity" template found in selected_paths, skipping.')
    
    pv_orientation = base_info.get("personal_values", {}).get("values_orientation", "")
    if not isinstance(pv_orientation, str):
        pv_orientation = json.dumps(pv_orientation, ensure_ascii=False)
        
    core_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Personal Values (for reference):\n" + pv_orientation + "\n\n"
        "Instructions: Based on the `life_story` and other information provided above, **develop and elaborate on** the 'Core Values, Beliefs, and Philosophy' section in English. Your aim is to distill and articulate the core values, beliefs, and philosophical outlook that are **evident or can be reasonably inferred from the `life_story` and other provided information**. Offer fresh insights by providing a **deeper, more nuanced interpretation or by highlighting connections within the provided data** that illuminate these guiding principles. Ensure that this elaboration is **logically consistent with and directly stems from** the provided information. **Do not introduce new values, beliefs, or philosophies that are not grounded in or clearly supported by the source material.** The section should be an insightful and coherent expansion of what can be understood from the source material.IMPORTANT: Avoid including anything related to community-building activities.Prohibit the use of words such as' balance 'and' balance '"
    )
    core_template = selected_paths.get("Core Values, Beliefs, and Philosophy")
    if core_template and core_template != "":
        print('Generating Core Values, Beliefs, and Philosophy...')
        core_values_section = generate_category_attributes(core_template, core_input, "Core Values, Beliefs, and Philosophy")
        nested_result = {}
        for path, value in core_values_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Core Values, Beliefs, and Philosophy":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Core Values, Beliefs, and Philosophy"] = nested_result
    else:
        print('No valid "Core Values, Beliefs, and Philosophy" template found in selected_paths, skipping.')
    
    life_attitude = base_info["life_attitude"]
    lifestyle_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Life Attitude (for reference):\n" + life_attitude + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `life_story`, `life_attitude`, and other information provided above, generate detailed Lifestyle and Daily Routine section in English. Use the life story to inform realistic daily routines that align with the person's experiences and background.Prohibit the use of words such as' balance 'and' balance '"
    )
    lifestyle_template = selected_paths.get("Lifestyle and Daily Routine")
    if lifestyle_template and lifestyle_template != "":
        print('Generating Lifestyle and Daily Routine...')
        lifestyle_section = generate_category_attributes(lifestyle_template, lifestyle_input, "Lifestyle and Daily Routine")
        nested_result = {}
        for path, value in lifestyle_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Lifestyle and Daily Routine":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Lifestyle and Daily Routine"] = nested_result
    else:
        print('No valid "Lifestyle and Daily Routine" template found in selected_paths, skipping.')
    
    cultural_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Life Attitude (for reference):\n" + life_attitude + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Lifestyle (for reference):\n" + json.dumps(profile.get("Lifestyle and Daily Routine", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `life_story`, `life_attitude`, and other information provided above, generate detailed Cultural and Social Context section in English. Use the life story to inform realistic cultural contexts that align with the person's experiences and background.Prohibit the use of words such as' balance 'and' balance '"
    )
    cultural_template = selected_paths.get("Cultural and Social Context")
    if cultural_template and cultural_template != "":
        print('Generating Cultural and Social Context...')
        cultural_section = generate_category_attributes(cultural_template, cultural_input, "Cultural and Social Context")
        nested_result = {}
        for path, value in cultural_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Cultural and Social Context":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Cultural and Social Context"] = nested_result
    else:
        print('No valid "Cultural and Social Context" template found in selected_paths, skipping.')
    
    interests = base_info["interests"]
    hobbies_input = (
        "Base Information (for reference):\n" + json.dumps(base_info, ensure_ascii=False, indent=2) + "\n\n"
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Demographic Information (for reference):\n" + json.dumps(profile.get("Demographic Information", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Career Information (for reference):\n" + json.dumps(profile.get("Career and Work Identity", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Core Values, Beliefs, and Philosophy (for reference):\n" + json.dumps(profile.get("Core Values, Beliefs, and Philosophy", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Lifestyle and Daily Routine (for reference):\n" + json.dumps(profile.get("Lifestyle and Daily Routine", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Cultural and Social Context (for reference):\n" + json.dumps(profile.get("Cultural and Social Context", {}), ensure_ascii=False, indent=2) + "\n\n"
        "Ensure that all hobbies, interests, and lifestyle choices presented are:1.  **Firmly anchored to and primarily derived from the hobbies indicated in `base_info` and experiences from `life_story`.**2.  Logically consistent with all provided information.3.  Enriched by supplementary information where appropriate, without overshadowing the core hobbies from `base_info`.**Do not introduce new primary hobbies or interests that are not clearly supported by or cannot be reasonably inferred from the `base_info` and `life_story` themselves.** Any lifestyle elements should logically flow from or align with these established hobbies and the overall profile.Prohibit the use of words such as' balance 'and' balance '"
    )
    hobbies_template = selected_paths.get("Hobbies, Interests, and Lifestyle")
    if hobbies_template and hobbies_template != "":
        print('Generating Hobbies, Interests, and Lifestyle...')
        hobbies_section = generate_category_attributes(hobbies_template, hobbies_input, "Hobbies, Interests, and Lifestyle")
        nested_result = {}
        for path, value in hobbies_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Hobbies, Interests, and Lifestyle":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Hobbies, Interests, and Lifestyle"] = nested_result
    else:
        print('No valid "Hobbies, Interests, and Lifestyle" template found in selected_paths, skipping.')
    
    other_attributes_input = (
        "Life Story (for reference):\n" + str(life_story) + "\n\n"
        "Complete Profile (for reference):\n" + json.dumps(profile, ensure_ascii=False, indent=2) + "\n\n"
        "Instructions: Based on the `life_story` and complete profile, generate the remaining attributes for the user profile in English with refined details. Ensure that all attributes are consistent with the person's life experiences as described in the life story."
    )
    other_template = selected_paths.get("Other Attributes")
    if other_template and other_template != "":
        print('Generating Other Attributes...')
        other_attributes_section = generate_category_attributes(other_template, other_attributes_input, "Other Attributes")
        nested_result = {}
        for path, value in other_attributes_section.items():
            parts = path.split('.')
            if len(parts) > 1 and parts[0] == "Other Attributes":
                parts = parts[1:]
            
            current = nested_result
            for i, part in enumerate(parts):
                if i == len(parts) - 1:
                    current[part] = value
                else:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
        profile["Other Attributes"] = nested_result
    else:
        print('No valid "Other Attributes" template found in selected_paths, skipping.')

    profile_for_summary = profile.copy()
    for key in ['base_info', 'Base Info', 'personal_story', 'interests', 'Occupations']:
         profile_for_summary.pop(key, None)
    
    final_summary_text = generate_final_summary(profile_for_summary, base_info)
    profile["Summary"] = final_summary_text
    
    for key in ['base_info', 'Base Info', 'personal_story', 'interests', 'Occupations']:
         profile.pop(key, None)

    return profile

def generate_multiple_profiles(num_rounds: int = 8) -> None:
    """Generates multiple rounds of user profiles across different attribute scales and saves them to a consolidated JSON.
    
    Args:
        num_rounds: The number of rounds to generate. Defaults to 8.
    """
    start_time = time.time()
    print(f"Starting {num_rounds} rounds of profile generation...")
    
    project_root = get_project_root()
    deeppersona_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(deeppersona_root, '..', '..'))
    output_dir = os.path.join(project_root, "deeppersona_ai", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    attribute_counts = [100, 150, 200, 250, 300, 350]
    total_profiles = num_rounds * len(attribute_counts)
    
    all_profiles = {
        "metadata": {
            "profiles_completed": 0,
            "total_profiles": total_profiles,
            "total_rounds": num_rounds,
            "description": "Collection of user profiles across multiple rounds and attribute counts."
        }
    }
    
    base_all_profiles_path = os.path.join(output_dir, "profile_ind.json")
    all_profiles_path = base_all_profiles_path
    
    actual_path = save_json_file(all_profiles_path, all_profiles, use_timestamp=False)
    print(f"Initialized merged file: {actual_path}")
    all_profiles_path = actual_path 
    
    profile_count = 0
    
    for round_num in range(num_rounds):
        print(f"\n===== Round {round_num+1}/{num_rounds} =====\n")
        
        for attr_index, current_attribute_count in enumerate(attribute_counts):
            profile_count += 1
            
            print(f"\n----- Generating Profile {round_num+1}.{attr_index+1} ({current_attribute_count} attributes) -----\n")
            
            try:
                profile = generate_single_profile(None, profile_count-1, current_attribute_count)
                
                if not profile:
                    print(f"Profile {round_num+1}.{attr_index+1} failed to generate, skipping.")
                    continue
                
                profile_key = f"Profile_R{round_num+1}_A{attr_index+1}_Count_{current_attribute_count}"
                all_profiles[profile_key] = profile
                all_profiles["metadata"]["profiles_completed"] = profile_count
                
                save_json_file(all_profiles_path, all_profiles, use_timestamp=False)
                print(f"\nProgress: {profile_count}/{total_profiles} profiles completed (Round {round_num+1}/{num_rounds}).")
                print(f"Profile appended to: {all_profiles_path}")
                print("\n" + "-"*50 + "\n")
            except Exception as e:
                print(f"Error generating Profile {round_num+1}.{attr_index+1}: {e}")
                continue
        
        print(f"\n===== Round {round_num+1}/{num_rounds} Completed =====\n")
        print("\n" + "="*50 + "\n")
    
    all_profiles["metadata"]["status"] = "completed"
    save_json_file(all_profiles_path, all_profiles, use_timestamp=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nSuccessfully generated {all_profiles['metadata']['profiles_completed']} profiles. Saved to: {all_profiles_path}")
    print(f"Generation completed in {elapsed_time:.2f} seconds.")

    profile_list = []
    for key, value in all_profiles.items():
        if key.startswith("Profile_"):
            profile_list.append(value)
    deeppersonal_path = os.path.join(project_root, "deeppersona_ai", "deeppersonal_agents.json")
    with open(deeppersonal_path, 'w', encoding='utf-8') as f:
        json.dump(profile_list, f, ensure_ascii=False, indent=2)
    print(f"List format saved to: {deeppersonal_path} ({len(profile_list)} profiles)")

if __name__ == "__main__":
    generate_multiple_profiles(3)
