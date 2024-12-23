import re

def extract_unique_tags(xml_path, output_file):
    # Regular expression to match tags (starting with '<' and ending with '>', excluding closing tags like '</...>')
    tag_pattern = r'<(?!/)[^>]+>'  # This excludes tags starting with '</'

    # Set to store unique tags
    unique_tags = set()
    
    # Read the XML file
    with open(xml_path, 'r', encoding='utf-8') as file:
        content = file.read()
        
        # Find all tags (opening and self-closing)
        tags = re.findall(tag_pattern, content)
        
        # Add each tag to the set (automatically ensures uniqueness)
        for tag in tags:
            unique_tags.add(tag.strip())  # Strip any extra whitespace
    
    # Write the unique tags to the output file
    with open(output_file, 'w', encoding='utf-8') as output:
        for tag in sorted(unique_tags):  # Sort the tags before writing for easier readability
            output.write(f"{tag}\n")
    
    print(f"Unique tags have been written to {output_file}")

xml_path = 'output_beautified_document.xml'  # Replace with your XML file path
output_file = 'tags.txt'
extract_unique_tags(xml_path, output_file)