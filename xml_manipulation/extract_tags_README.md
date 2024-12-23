# XML Unique Tag Extractor

This script reads an XML file, extracts all unique opening tags (including self-closing tags but excluding closing tags), and writes them to an output file. It ensures that each tag appears only once in the output file, providing a list of unique tags used within the XML structure.

## Overview

The script performs the following steps:
1. Reads the XML file specified by `xml_path`.
2. Uses a regular expression to find all unique opening and self-closing tags.
3. Excludes any closing tags (e.g., `</tag>`).
4. Writes each unique tag to a specified output file, sorted for easier readability.

## Requirements

- Python 3.x

## Code Explanation

### `extract_unique_tags(xml_path, output_file)`

This function extracts unique tags from an XML file and saves them in an output text file.

```python
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
