#!/usr/bin/env python3
import os
import json
from tqdm import tqdm
from typing import Tuple, List
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

OPENAI_API_KEY = "OPENAI_API_KEY"
GPT_MODEL = "gpt-4o"

client = OpenAI(
    api_key=OPENAI_API_KEY,
)

def get_sibling_paths(data: dict, current_path: str) -> List[str]:
    """Gets sibling paths for a given path."""
    if not current_path:
        return []
        
    parts = current_path.split('.')
    parent_path = '.'.join(parts[:-1]) 
    
    if not parent_path:
        return [key for key in data.keys() if key != parts[0]]
    
    current_dict = data
    for part in parent_path.split('.'):
        current_dict = current_dict.get(part, {})
    
    siblings = [f"{parent_path}.{key}" for key in current_dict.keys() if key != parts[-1]]
    return siblings

def convert_tree_to_paths(data: dict, current_path: str = "") -> list:
    """Converts a tree structure to a list of paths."""
    paths = []
    for key, value in data.items():
        new_path = f"{current_path}.{key}" if current_path else key
        if isinstance(value, dict):
            if not value: 
                paths.append(new_path)
            else:
                paths.extend(convert_tree_to_paths(value, new_path))
    return paths

def convert_paths_to_tree(paths: list) -> dict:
    """Converts a list of paths back to a tree structure."""
    tree = {}
    for path in paths:
        current = tree
        parts = path.split('.')
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                current[part] = {} 
            else:
                if part not in current:
                    current[part] = {}
                current = current[part]
    return tree

def remove_duplicates(paths: list) -> list:
    """Removes duplicate paths while preserving order."""
    if not paths:
        return []
    
    result = [paths[0]]
    for i in range(1, len(paths)):
        if paths[i] != paths[i-1]:
            result.append(paths[i])
    return result

def check_path_similarity(path1: str, path2: str) -> bool:
    """
    Checks if two paths are too similar using sentence-transformers.
    """
    embeddings = model.encode([path1, path2])
    
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    threshold = 0.85 
    if similarity > threshold:
        print(f"Found similar paths (similarity: {similarity:.2f}):\n- {path1}\n- {path2}")
    return similarity > threshold

def check_level_compatibility(current_level: str, parent_level: str) -> bool:
    """
    Checks if the current level is compatible with its parent level using GPT.
    """
    prompt = f"""Analyze if the current level '{current_level}' is compatible with its parent level '{parent_level}'.

Rules:
1. The current level must be a logical subdivision, attribute, or subcategory of the parent level.
2. Both levels should be general categories, not specific instances.
3. The current level should represent a more specific subset of the parent level.
4. The relationship between levels must make logical sense in a hierarchical structure.

Please respond with ONLY 'true' or 'false'.
- 'true' means the levels are compatible and form a valid hierarchy
- 'false' means the levels are incompatible or illogical
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a path hierarchy analyzer. Your task is to strictly judge if two levels are compatible. Only return 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.lower() == 'true'
        if result:
            print(f"Level compatibility check (GPT):\n- Current level: {current_level}\n- Parent level: {parent_level}")
        else:
            print(f"Level incompatibility found (GPT):\n- Current level: {current_level}\n- Parent level: {parent_level}")
        return result
        
    except Exception as e:
        print(f"Error calling GPT API: {e}")
        return False

def check_attribute_quality(attribute: str, full_path: str) -> bool:
    """
    Checks if an attribute meets the quality requirements for personalization.
    """
    prompt = f"""Analyze if the attribute '{attribute}' (from path: {full_path}) meets these requirements:

1. User-Centric Focus:
   - Must describe personal characteristics/attributes
   - Should be general enough to apply to many individuals
   - Should enable rich content generation about a person

2. Category Requirements:
   - Must be a general category (no specific instances, behaviors, or values)

Please respond with ONLY 'true' or 'false'.
- 'true' means the attribute meets ALL requirements
- 'false' means it fails one or more requirements
"""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a strict attribute quality checker for persona generation. Your task is to ensure attributes meet specific quality standards. Only return 'true' or 'false'."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.lower() == 'true'
        
        if result:
            print(f"Retained: {full_path}")
        
        return result
        
    except Exception as e:
        print(f"Error checking attribute quality: {e}")
        return False

def validate_path_levels(path: str) -> bool:
    """Validates each node and level structure of the path."""
    levels = path.split('.')
    
    if not check_attribute_quality(levels[-1], path):
        return False
    
    for i in range(len(levels)-1, 0, -1):
        current = levels[i] 
        parent = levels[i-1] 
        if not check_level_compatibility(current, parent):
            print(f"Level incompatibility in path '{path}': '{current}' is not compatible with parent '{parent}'")
            return False
            
    return True

class PathFilter:
    def __init__(self):
        self.retained_paths = [] 
    
    def check_node_quality(self, node: str, path: str) -> bool:
        """Checks if a node meets quality requirements."""
        return check_attribute_quality(node, path)
    
    def check_node_compatibility(self, current: str, parent: str) -> bool:
        """Checks if a node is compatible with its parent."""
        return check_level_compatibility(current, parent)
    
    def filter_tree(self, data: dict, current_path: str = "", pbar=None) -> dict:
        """
        Processes the tree structure in two phases:
        1. First check path similarity within first-level groups
        2. Then process leaf nodes with quality and compatibility checks
        """
        print("\nPhase 1: Checking path similarity...")
        paths = convert_tree_to_paths(data)
        paths.sort()
        paths = remove_duplicates(paths)
        
        paths_by_first_level = {}
        first_level_paths = [] 
        
        for path in paths:
            parts = path.split('.')
            if len(parts) <= 1: 
                first_level_paths.append(path)
                continue
            
            first_level = parts[0]
            if first_level not in paths_by_first_level:
                paths_by_first_level[first_level] = []
            paths_by_first_level[first_level].append(path)
        
        filtered_paths = first_level_paths.copy()
        
        for first_level, group_paths in paths_by_first_level.items():
            group_filtered_paths = []
            for path in group_paths:
                if pbar:
                    pbar.update(1)
                
                is_similar = False
                for retained_path in group_filtered_paths:
                    if check_path_similarity(path, retained_path):
                        is_similar = True
                        print(f"Found similar paths:\n- {path}\n- {retained_path}")
                        break
                
                if not is_similar:
                    group_filtered_paths.append(path)
                    print(f"Passed similarity check: {path}")
            
            filtered_paths.extend(group_filtered_paths)
        
        print("\nPhase 2: Processing leaf nodes...")
        def process_leaf_nodes(tree_data: dict, path: str = "") -> dict:
            filtered = {}
            
            for key, value in list(tree_data.items()):
                new_path = f"{path}.{key}" if path else key
                
                if new_path not in filtered_paths and new_path not in first_level_paths:
                    continue
                
                levels = new_path.split('.')
                
                if len(levels) > 1:
                    if not self.check_node_quality(levels[-1], new_path):
                        continue
                    
                    if not self.check_node_compatibility(levels[-1], levels[-2]):
                        continue
                
                if isinstance(value, dict) and not value:
                    filtered[key] = {}
                    self.retained_paths.append(new_path)
                    print(f"Retained leaf node: {new_path}")
                
                elif isinstance(value, dict):
                    filtered_children = process_leaf_nodes(value, new_path)
                    if filtered_children or len(levels) == 1: 
                        filtered[key] = filtered_children
            
            return filtered
        
        return process_leaf_nodes(data)

def count_leaves(d: dict) -> int:
    """Counts the number of leaf nodes in a dictionary."""
    count = 0
    for value in d.values():
        if isinstance(value, dict):
            if not value: 
                count += 1
            else:
                count += count_leaves(value)
    return count

def get_all_paths(d: dict, current_path: str = "") -> List[str]:
    """Retrieves all paths from a dictionary."""
    paths = []
    for key, value in d.items():
        new_path = f"{current_path}.{key}" if current_path else key
        if isinstance(value, dict):
            if not value: 
                paths.append(new_path)
            else:
                paths.extend(get_all_paths(value, new_path))
    return paths

def main():
    print("Starting processing...")
    input_file = "/home/zhou/persona/src/process_attributes_test/2.24/outputs/run_20250326_125810/attributes_merged.json"
    output_file = os.path.join(os.path.dirname(input_file), "filtered_attributes1.json")
    log_file = os.path.join(os.path.dirname(input_file), "filter_log1.txt")
    
    try:
        print("Reading input file...")
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    original_leaves = count_leaves(data)
    all_paths = get_all_paths(data)
    print(f"Starting to filter {original_leaves} leaf nodes...")
    
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Start file: {os.path.basename(input_file)}\n")
        f.write(f"Original leaf nodes: {original_leaves}\n\n")
    
    path_filter = PathFilter()
    with tqdm(total=original_leaves, desc="Filtering nodes") as pbar:
        filtered_data = path_filter.filter_tree(data, pbar=pbar)
    
    filtered_leaves = count_leaves(filtered_data)
    filtered_paths = get_all_paths(filtered_data)
    removed_paths = set(all_paths) - set(filtered_paths)
    similar_paths = set(all_paths) - set(filtered_paths) - set(path_filter.retained_paths)

    try:
        print("\nSaving results...")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write("\nRetained paths:\n")
            for path in sorted(path_filter.retained_paths):
                f.write(f"+ {path}\n")
                
            f.write("\nRemoved paths:\n")
            for path in sorted(removed_paths):
                if path in similar_paths:
                    f.write(f"- {path} (Similar to other paths)\n")
                else:
                    f.write(f"- {path} (Failed requirements)\n")
                    
            f.write(f"\nStatistics:\n")
            f.write(f"- Original leaf nodes: {original_leaves}\n")
            f.write(f"- Filtered leaf nodes: {filtered_leaves}\n")
            f.write(f"- Removed nodes: {original_leaves - filtered_leaves}\n")
            f.write(f"- Similar paths removed: {len(similar_paths)}\n")
            
        print(f"Done! Results saved to: {output_file}")
        print(f"Log saved to: {log_file}")
        print(f"\nStatistics:")
        print(f"- Original leaf nodes: {original_leaves}")
        print(f"- Filtered leaf nodes: {filtered_leaves}")
        print(f"- Removed nodes: {original_leaves - filtered_leaves}")
        print(f"- Similar paths removed: {len(similar_paths)}")
        print(f"- Total removal rate: {((original_leaves - filtered_leaves) / original_leaves * 100):.2f}%")
    except Exception as e:
        print(f"Failed to save file: {e}")

if __name__ == "__main__":
    main()