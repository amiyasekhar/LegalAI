import re
from collections import OrderedDict

def convert_to_string(file_path):
    # Open the XML file and read its content into a string
    with open(file_path, 'r', encoding='utf-8') as file:
        xml_string = file.read()
    
    return xml_string

def extract_wp_tags(xml_string):
    # Regular expression to match everything within <w:p> ... </w:p>
    wp_tags = re.findall(r'<w:p[\s\S]*?</w:p>', xml_string)
    
    return wp_tags

file_path = 'output_beautified_document.xml'  # Replace with the path to your XML file
xml_string = convert_to_string(file_path)

# Extract <w:p> ... </w:p> tags into an ordered list
wp_tag_list = extract_wp_tags(xml_string)

'''
# Print the ordered list of <w:p> ... </w:p> tags
for i, wp_tag in enumerate(wp_tag_list, start=1):
    print(f"Tag {i}: {wp_tag}\n")
'''


def process_paragraphs(wp_tags):
    # Initialize an ordered list for storing refined components
    refined_components = [wp_tags[0]]

    # Regex to extract text from <w:t> tags
    w_t_pattern = re.compile(r'<w:t[^>]*>(.*?)</w:t>', re.DOTALL)

    # Skip the first element and iterate through <w:p> elements
    for i in range(1, len(wp_tags)):
        current_p = wp_tags[i]

        # Find all <w:t> tags in the current <w:p> element
        w_t_matches = w_t_pattern.findall(current_p)

        # Check if there are no <w:t> children (no text in the paragraph)
        if not w_t_matches:
            # Add the whole <w:p> element followed by "\n"
            refined_components.append(current_p)
            refined_components.append("\n")
            continue
        
        # Get the last <w:t> child text
        last_w_t_text = w_t_matches[-1].strip() if w_t_matches else ""

        # If the last character is not a period, group paragraphs
        if last_w_t_text and not last_w_t_text.endswith('.'):
            grouped_paragraph = current_p  # Start grouping with the current paragraph
            for j in range(i + 1, len(wp_tags)):
                next_p = wp_tags[j]
                next_w_t_matches = w_t_pattern.findall(next_p)

                if next_w_t_matches:
                    last_w_t_text = next_w_t_matches[-1].strip()
                    # Append the next paragraph to the grouping
                    grouped_paragraph += next_p

                    if last_w_t_text.endswith('.'):
                        # Stop grouping when we hit a paragraph ending with "."
                        refined_components.append(grouped_paragraph)
                        i = j  # Move the outer loop to this paragraph
                        break
            else:
                # If we exhaust all paragraphs and none ends with ".", add the grouped content
                refined_components.append(grouped_paragraph)
        else:
            # Add the <w:p> element as is
            refined_components.append(current_p)
    
    return refined_components

refined_components = process_paragraphs(wp_tag_list)

# Write the output to a file or use it further
with open('refined_output.txt', 'w') as f:
    for component in refined_components:
        f.write(f"{component}\n-------------------")