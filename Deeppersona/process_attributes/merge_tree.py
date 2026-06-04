import json
import os
from openai import OpenAI
from typing import List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

OPENAI_API_KEY = "OPENAI_API_KEY"
GPT_MODEL = "gpt-4o"

class OutputManager:
    def __init__(self):
        """Initialize the output manager with an outputs directory in the current script's directory."""
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.outputs_dir = os.path.join(script_dir, "outputs")
        os.makedirs(self.outputs_dir, exist_ok=True)
        
        self.output_dir = os.path.join(self.outputs_dir, f"run_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Created output directory: {self.output_dir}")
    
    def get_output_path(self, filename: str) -> str:
        """Get the full path for an output file."""
        return os.path.join(self.output_dir, filename)

@dataclass
class TreeNode:
    value: str
    children: Dict[str, 'TreeNode']
    level: int
    original_path: str
    
    def __init__(self, value: str, level: int, original_path: str = ""):
        self.value = value
        self.children = {}
        self.level = level
        self.original_path = original_path
        
    def __hash__(self):
        return hash(self.original_path)
        
    def __eq__(self, other):
        if not isinstance(other, TreeNode):
            return False
        return self.original_path == other.original_path

def json_to_tree(json_data: Dict, current_key: str = "root", level: int = 0, path: str = "") -> TreeNode:
    """Converts JSON data into a TreeNode object."""
    node = TreeNode(value=current_key, level=level, original_path=path)
    
    if isinstance(json_data, dict):
        for key, value in sorted(json_data.items()):
            new_path = f"{path}.{key}" if path else key
            child_node = json_to_tree(value, key, level + 1, new_path)
            node.children[key] = child_node
    
    return node

def tree_to_json(node: TreeNode) -> Dict:
    """Converts a TreeNode back to a JSON format with a consistent structure."""
    result = {}
    
    for key, child in sorted(node.children.items()):
        result[key] = tree_to_json(child)
        
    return result

def get_nodes_at_level(root: TreeNode, target_level: int) -> List[TreeNode]:
    """Retrieves all nodes at a specific depth level."""
    if target_level == root.level:
        return [root]
    nodes = []
    for child in root.children.values():
        nodes.extend(get_nodes_at_level(child, target_level))
    return nodes

def validate_gpt_response(response_text: str) -> Dict[str, str]:
    """Validates and cleans up the GPT response to ensure it is valid JSON."""
    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        try:
            json_str = response_text[response_text.find('{'):response_text.rfind('}')+1]
            return json.loads(json_str)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse GPT response: {response_text}")
            print(f"Error: {str(e)}")
            return {}

def process_merge_response(mapping: Dict[str, str], nodes_to_merge: List[TreeNode]) -> Dict[str, TreeNode]:
    """Processes the merge mapping and creates the merged nodes."""
    if not mapping:
        return {node.value: node for node in nodes_to_merge}
        
    print(f"Processing mapping: {mapping}")
    
    new_nodes = {}
    for old_node in nodes_to_merge:
        new_value = mapping.get(old_node.value, old_node.value)
        
        if new_value not in new_nodes:
            if new_value == old_node.value:
                new_nodes[new_value] = old_node
            else:
                new_node = TreeNode(
                    new_value,
                    old_node.level,
                    new_value
                )
                new_nodes[new_value] = new_node
        
        if new_value != old_node.value:
            new_nodes[new_value].children.update(old_node.children)
    
    return new_nodes

def merge_level_nodes(nodes: List[TreeNode], level: int, client: OpenAI) -> Dict[str, TreeNode]:
    """Merges nodes at the same level based on personalization and abstraction requirements."""
    if len(nodes) <= 1:
        return {node.value: node for node in nodes}

    try:
        prompt = f"""You are an expert in analyzing and organizing hierarchical data structures.
Your task is to analyze nodes at the same level and suggest merges based on semantic similarity.
Return ONLY a JSON dictionary mapping current node names to new names, nothing else.

Current nodes at level {level}: {[n.value for n in nodes]}

Merging Strategy:
1. Primary Goal: Merge semantically similar attributes

2 Similarity Thresholds:
   - If nodes share core concept/purpose (>80% similar): Directly merge
   - If completely different (<80% similar): Keep separate

STRICT REQUIREMENTS:
1. User-Centric Focus:
   - Must be user personalization attributes that reflect individual characteristics/attributes
  
2. Must be general category (no specific instances, behaviors, or values)

3. Must logically refine parent level
  
4. Attributes must be highly general, enabling GPT to generate rich content for that attribute
"""

        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a JSON-only response bot. Return valid JSON dictionaries only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        mapping = validate_gpt_response(response.choices[0].message.content.strip())
        return process_merge_response(mapping, nodes)
        
    except Exception as e:
        print(f"Error in merge_level_nodes at level {level}: {str(e)}")
        return {node.value: node for node in nodes}

def validate_parent_attribute(node: TreeNode, client: OpenAI) -> str:
    """Validates and potentially updates the parent attribute name to ensure it meets requirements."""
    if not node.original_path or '.' not in node.original_path:
        return node.value
        
    parent_path = '.'.join(node.original_path.split('.')[:-1])
    
    prompt = f"""Analyze this attribute name and determine if it meets these requirements:
1. User-Centric Focus:
   - Must be user personalization attributes that reflect individual characteristics/attributes
  
2. Check each level:
  - Must be general category (no specific instances, behaviors, or values)
  - Must logically refine parent level

3. Clarity and Meaningfulness:
   - MUST be clear and meaningful in describing personality traits

Attribute to analyze: {parent_path}

If the attribute meets ALL requirements, respond with: KEEP:{parent_path}
If it doesn't meet requirements, suggest a new name that does meet ALL requirements with: CHANGE:new_name

Provide ONLY the response in the format above, no other text."""

    try:
        response = client.chat.completions.create(
            model=GPT_MODEL,
            messages=[
                {"role": "system", "content": "You are a direct response bot. Respond only with KEEP: or CHANGE: followed by the attribute name."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        result = response.choices[0].message.content.strip()
        
        if result.startswith('KEEP:'):
            return parent_path
        elif result.startswith('CHANGE:'):
            new_name = result.split(':', 1)[1].strip()
            return new_name
        else:
            print(f"Unexpected response format: {result}")
            return parent_path
            
    except Exception as e:
        print(f"Error in validate_parent_attribute: {str(e)}")
        return parent_path

def process_tree_level_by_level(root: TreeNode, client: OpenAI, output_manager: OutputManager) -> None:
    """Processes the tree level by level while preserving first-level attributes."""
    max_level = 4
    
    for first_level_key, first_level_node in list(root.children.items()):
        new_children = {}
        for child_key, child_node in first_level_node.children.items():
            if '.' in child_key:
                prefix, rest = child_key.split('.', 1)
                if prefix == first_level_key:
                    child_node.value = rest
                    child_node.original_path = f"{first_level_key}.{rest}"
                    new_children[rest] = child_node
            else:
                new_children[child_key] = child_node
        first_level_node.children = new_children
    
    save_intermediate_results(root, 1, output_manager)
    
    print("\nProcessing Level 2...")
    for first_level_key, first_level_node in root.children.items():
        if first_level_node.children:
            print(f"\nProcessing Level 2 nodes under '{first_level_key}'...")
            new_nodes = merge_level_nodes(list(first_level_node.children.values()), 2, client)
            first_level_node.children = new_nodes
            
            for node in new_nodes.values():
                new_parent = validate_parent_attribute(node, client)
                if new_parent != first_level_key:
                    node.original_path = f"{new_parent}.{node.value}"
    
    save_intermediate_results(root, 2, output_manager)
    
    print("\nProcessing Level 3...")
    for x_node in root.children.values():
        for y_key, y_node in list(x_node.children.items()):
            if y_node.children:
                print(f"\nProcessing Level 3 nodes under '{x_node.value}.{y_key}'...")
                new_nodes = merge_level_nodes(list(y_node.children.values()), 3, client)
                y_node.children = new_nodes
                
                parent_path = f"{x_node.value}.{y_key}"
                for node in new_nodes.values():
                    new_parent = validate_parent_attribute(node, client)
                    if new_parent != parent_path:
                        node.original_path = f"{new_parent}.{node.value}"
    
    save_intermediate_results(root, 3, output_manager)
    
    print("\nChecking Level 4...")
    has_fourth_level = False
    for x_node in root.children.values():
        for y_node in x_node.children.values():
            for z_key, z_node in list(y_node.children.items()):
                if z_node.children:
                    has_fourth_level = True
                    break
            if has_fourth_level:
                break
        if has_fourth_level:
            break
    
    if has_fourth_level:
        print("Level 4 nodes found. Processing...")
        for x_node in root.children.values():
            for y_node in x_node.children.values():
                for z_key, z_node in list(y_node.children.items()):
                    if z_node.children:
                        print(f"\nProcessing nodes under '{x_node.value}.{y_node.value}.{z_key}'...")
                        new_nodes = merge_level_nodes(list(z_node.children.values()), 4, client)
                        z_node.children = new_nodes
        save_intermediate_results(root, 4, output_manager)
        print("Level 4 processing complete.")
    else:
        print("No Level 4 nodes found. Skipping.")
    
    print("\nAll levels processed successfully.")

def save_intermediate_results(root: TreeNode, level: int, output_manager: OutputManager) -> None:
    """Saves intermediate results in the same format as the input JSON."""
    output_path = output_manager.get_output_path(f"attributes_level_{level}.json")
    json_data = tree_to_json(root)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    print(f"Saved level {level} results to {output_path}")

def build_simple_tree_structure(attributes: list) -> dict:
    """
    Converts a list of dot-separated attributes into a nested dictionary tree structure.
    
    Args:
        attributes: List of dot-separated attributes, e.g., ["X.Y.Z", "X.Y2.Z2"]
        
    Returns:
        dict: Nested tree structure dictionary.
    """
    tree = {}
    for attr in attributes:
        parts = attr.split('.')
        current = tree
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    
    return tree

def main():
    input_file = "PATH.json"
    output_manager = OutputManager()
    
    try:
        print("Building and optimizing tree structure...")
        with open(input_file, 'r', encoding='utf-8') as f:
            attributes = json.load(f)
        
        tree = build_simple_tree_structure(attributes)
        
        initial_tree_path = output_manager.get_output_path("initial_tree.json")
        with open(initial_tree_path, 'w', encoding='utf-8') as f:
            json.dump(tree, f, indent=2, ensure_ascii=False)
        print(f"Saved initial tree structure to {initial_tree_path}")
        
        print("Converting to TreeNode structure...")
        root = TreeNode(value="root", level=0)
        for key, value in sorted(tree.items()):
            child_node = json_to_tree(value, key, level=1, path=key)
            root.children[key] = child_node
        print(f"Successfully loaded tree with {len(root.children)} top-level nodes")
        
    except Exception as e:
        print(f"Error building tree structure: {str(e)}")
        return
    
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    print("\nStarting tree merge process...")
    process_tree_level_by_level(root, client, output_manager)
    
    print("Saving final results...")
    final_json = tree_to_json(root)
    output_path = output_manager.get_output_path("attributes_merged.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_json, f, indent=2, ensure_ascii=False)
    print(f"Final merged attributes saved to {output_path}")

if __name__ == "__main__":
    main()