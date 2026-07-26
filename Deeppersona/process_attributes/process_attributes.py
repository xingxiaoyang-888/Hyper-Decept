import json

# Read the JSON file
with open('PATH.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# Get personalized_attributes and remove duplicates
personalized_attributes = list(set(data['personalized_attributes']))

# Sort alphabetically
personalized_attributes.sort()

# Print results
print(f"Number of attributes before deduplication: {len(data['personalized_attributes'])}")
print(f"Number of attributes after deduplication: {len(personalized_attributes)}")
print("\nDeduplicated list of attributes:")
for attr in personalized_attributes:
    print(attr)