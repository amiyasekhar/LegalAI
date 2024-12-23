import xml.etree.ElementTree as ET
from copy import deepcopy
import os

# Define stop words for the demo (if applicable)
stop_words = ['END', 'STOP']

# Initializing paragraph properties (can be expanded as needed)
para_properties_template = {
    "alignment": None,
    "indent": {
        "left": None,
        "right": None,
        "firstLine": None,
        "hanging": None
    },
    "spacing": {
        "before": None,
        "after": None,
        "line": None,
        "lineRule": None
    },
    "borders": {
        "top": {"val": None, "size": None, "color": None},
        "left": {"val": None, "size": None, "color": None},
        "bottom": {"val": None, "size": None, "color": None},
        "right": {"val": None, "size": None, "color": None}
    },
    "shading": {
        "val": None,
        "color": None,
        "fill": None
    },
    "numbering": {
        "ilvl": None,
        "numId": None
    },
    "textDirection": None,
    "outlineLevel": None,
    "pageBreakBefore": None,
    "keepLinesTogether": None,
    "keepWithNext": None,
    "widowControl": None,
    "paragraphStyle": None,
    "frameProperties": {
        "width": None,
        "height": None,
        "hSpace": None,
        "vSpace": None
    },
    "suppressLineNumbers": None,
    "textAlignment": None,
    "suppressAutoHyphens": None,
    "contextualSpacing": None,
    "divId": None,
    "mirrorIndents": None,
    "textDirectionVertical": None,
    "dropCap": None,
    "tabs": {
        "position": None,
        "val": None,
        "leader": None
    }
}

# Define namespaces
namespaces = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'mc': 'http://schemas.openxmlformats.org/markup-compatibility/2006',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'c': 'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    # Add other namespaces as needed
}

def strip_namespace(tag):
    """Strip the namespace from the XML tag."""
    return tag.split('}', 1)[1] if '}' in tag else tag

def is_stop_word(text):
    """Check if the text contains any stop word."""
    return any(word in text.upper() for word in stop_words)

def get_text_recursive(element):
    """
    Recursively extract all <w:t> text from an XML element and its descendants.

    Args:
        element (xml.etree.ElementTree.Element): The XML element to traverse.

    Returns:
        list: A list of text strings extracted from <w:t> elements.
    """
    texts = []
    for elem in element.iter():
        if strip_namespace(elem.tag) == 't' and elem.text:
            texts.append(elem.text)
    return texts

def extract_text_from_body_children(xml_file):
    """
    Extract text from all direct children of <w:body> and their nested descendants.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        list: A list of dictionaries containing child index, tag, and extracted texts.
    """
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
    except ET.ParseError as pe:
        print(f"XML parsing error in '{xml_file}': {pe}")
        return []
    except Exception as e:
        print(f"Error opening '{xml_file}': {e}")
        return []

    body = root.find('w:body', namespaces)
    if body is None:
        print(f"No <w:body> found in '{xml_file}'.")
        return []

    extracted_texts = []
    child_index = 1  # To number the children

    debug_file_path = './xml_contract_parser_3_debug.txt'
    with open(debug_file_path, 'a', encoding='utf-8') as db:
        db.write(f"From: {xml_file}\n")
        for child in body:
            child_tag = strip_namespace(child.tag)
            child_texts = get_text_recursive(child)

            # Combine texts without stripping
            combined_text = ''.join(child_texts)

            # Handle cases where <w:p> has no <w:t>
            if not combined_text and child_tag == 'p':
                combined_text = "***THIS IS AN EMPTY PARA***"

            # Check if combined_text is non-empty or is a newline
            if combined_text and not is_stop_word(combined_text):
                extracted_texts.append({
                    f"Child {child_index} ({child_tag})": combined_text
                })
                db.write(f"Child {child_index} ({child_tag}): {combined_text}\n")
                child_index += 1  # Increment only when a child has text
            elif combined_text == "***THIS IS AN EMPTY PARA***" and child_tag == 'p':
                # Represent empty paragraph as newline
                extracted_texts.append({
                    f"Child {child_index} ({child_tag})": combined_text
                })
                db.write(f"Child {child_index} ({child_tag}): {combined_text}\n")
                child_index += 1

        db.write("\n")
    return extracted_texts

def process_xml_files(xml_directory):
    """
    Process all .xml files in the given directory, extract text, and save to .txt files.

    Args:
        xml_directory (str): Path to the directory containing XML files.
    """
    if not os.path.isdir(xml_directory):
        print(f"The directory '{xml_directory}' does not exist.")
        return

    files = os.listdir(xml_directory)

    for file_name in files:
        if file_name.lower().endswith('.xml'):
            xml_file_path = os.path.join(xml_directory, file_name)
            base_name = os.path.splitext(file_name)[0]
            output_txt_name = f"{base_name}-written-contract.txt"
            output_txt_path = os.path.join("/Users/amiyasekhar/CLM/MAIN_STUIFF_contract_parsing/results_3", output_txt_name)

            extracted_data = extract_text_from_body_children(xml_file_path)

            if extracted_data:
                try:
                    with open(output_txt_path, 'w', encoding='utf-8') as out_file:
                        for entry in extracted_data:
                            for child_key, child_text in entry.items():
                                out_file.write(f"{child_key}:\n{child_text}\n\n")
                    print(f"Written contract saved to '{output_txt_path}'.")
                except Exception as e:
                    print(f"Error writing to '{output_txt_path}': {e}")
            else:
                print(f"No relevant text extracted from '{xml_file_path}'.")

if __name__ == "__main__":
    # Directory containing the XML files
    xml_directory = '/Users/amiyasekhar/CLM/contracts'  # Update this path as needed

    # Process XML files
    process_xml_files(xml_directory)