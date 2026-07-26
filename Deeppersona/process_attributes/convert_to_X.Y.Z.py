#!/usr/bin/env python3
import os
import json
from collections import defaultdict

def extract_paths(data, prefix=""):
    """
    Recursively extracts all paths from a nested dictionary, separated by dots.
    If a node is empty or not a dictionary, it is considered a leaf node.
    """
    paths = []
    if isinstance(data, dict):
        if not data:
            if prefix:
                paths.append(prefix)
            return paths
        
        for key, value in data.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                child_paths = extract_paths(value, new_prefix)
                if child_paths:
                    paths.extend(child_paths)
                else:
                    paths.append(new_prefix)
            else:
                paths.append(new_prefix)
    else:
        if prefix:
            paths.append(prefix)
            
    return paths

def build_parent_child_map(paths):
    """
    Builds a parent-child mapping where each dot-separated part becomes an independent node.
    """
    parent_child_map = defaultdict(set)
    
    for path in paths:
        parts = path.split('.')
        current_path = ""
        
        for i, part in enumerate(parts):
            if current_path:
                current_path = f"{current_path}.{part}"
            else:
                current_path = part
                
            if i > 0:
                parent_path = '.'.join(parts[:i])
                parent_child_map[parent_path].add(current_path)
            
            if i == len(parts) - 1:
                parent_child_map[current_path] = set() 
                
    return {k: sorted(v) for k, v in parent_child_map.items()}

def generate_tree_text(parent_child_map):
    """Generates a text-based tree structure."""
    tree_lines = []
    
    all_children = set()
    for children in parent_child_map.values():
        all_children.update(children)
    root_nodes = set(parent_child_map.keys()) - all_children
    
    def add_node(node, prefix="", seen=None):
        if seen is None:
            seen = set()
            
        if node in seen:
            return
        seen.add(node)
        
        display_name = node.split('.')[-1]
        tree_lines.append(f"{prefix}- {display_name}")
        
        children = sorted(parent_child_map.get(node, []))
        for i, child in enumerate(children):
            is_last = i == len(children) - 1
            child_prefix = prefix + ('  └─ ' if is_last else '  ├─ ')
            next_prefix = prefix + ('  ' if is_last else '  │ ')
            add_node(child, next_prefix, seen)
            
    for root in sorted(root_nodes):
        add_node(root)
        tree_lines.append('')
        
    return tree_lines

def main():
    input_file = "PATH"   
    output_json = os.path.join(os.path.dirname(input_file), "X.Y.Z_3.6.json")
    output_txt = os.path.join(os.path.dirname(input_file), "X.Y.Z_3.6.txt")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to read file: {e}")
        return

    paths = extract_paths(data)
    result = {"paths": sorted(paths)}
    
    try:
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Converted paths saved to: {output_json}")
    except Exception as e:
        print(f"Failed to save JSON file: {e}")
        return
    
    try:
        parent_child_map = build_parent_child_map(paths)
        tree_lines = generate_tree_text(parent_child_map)
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            f.write('\n'.join(tree_lines))
        print(f"Tree text structure saved to: {output_txt}")
    except Exception as e:
        print(f"Failed to save tree text: {e}")

if __name__ == "__main__":
    main()