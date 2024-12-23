import xml.etree.ElementTree as ET
import os

# Define stop words for the demo
stop_words = ['END', 'STOP']

# Helper functions
def is_bold(run, namespaces):
    """Check if a run of text is bold."""
    return run.find('.//w:b', namespaces=namespaces) is not None

def is_uppercase(text):
    """Check if text is uppercase."""
    return text.isupper()

def is_stop_word(text):
    """Check if the text contains a stop word."""
    return any(word in text for word in stop_words)

def is_heading(text):
    """Check if text is a heading."""
    return text.strip().upper().startswith('SECTION')

# Function to extract headings and content from the XML contract
def extract_headings_and_content_from_xml(xml_file):
    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Define namespaces
    namespaces = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

    # Variables to store current state
    contract_title = ""
    is_title = False
    current_heading = ""
    current_text = ""
    heading_text = []
    stop_extraction = False

    # Iterate through paragraphs
    for paragraph_elem in root.findall('.//w:p', namespaces=namespaces):
        if stop_extraction:
            break  # Stop extraction if stop_extraction is True
        


        '''
        If this is the first paragraph of the whole document, this has to be the title. is_title = true

        Check if the para has any paragraph properties

        '''

        # Collect all text in the paragraph
        paragraph_text = ""
        for run_elem in paragraph_elem.findall('.//w:r', namespaces=namespaces):
            run_text_elem = run_elem.find('.//w:t', namespaces=namespaces)
            if run_text_elem is not None and run_text_elem.text:
                text = run_text_elem.text.strip()
                paragraph_text += text

        print("Paragraph text: ", paragraph_text)

        # Determine if this paragraph is a heading
        if is_heading(paragraph_text) or is_uppercase(paragraph_text):
            # It's a heading
            if current_heading and current_text:
                heading_text.append({current_heading: current_text.strip()})
                current_text = ""  # Reset content for the new heading
            current_heading = paragraph_text
        else:
            # It's content
            current_text += paragraph_text + " "  # Add space for clarity

        # Stop if a stop word is encountered
        if is_stop_word(paragraph_text):
            stop_extraction = True
            break

    # Append the last heading and text if any
    if current_heading and current_text:
        heading_text.append({current_heading: current_text.strip()})

    return heading_text

# Main function to test the extraction
if __name__ == "__main__":
    directory_path = '/Users/amiyasekhar/CLM/contracts'

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.xml'):
            # Path to the XML contract file (replace with your file path)
            xml_file = os.path.join(directory_path, file_name)

            # Extract headings and content
            extracted_data = extract_headings_and_content_from_xml(xml_file)
            
            # Define the output file name based on the XML file name
            output_file_path = os.path.join("/Users/amiyasekhar/CLM/MAIN_STUIFF_contract_parsing", f"{os.path.splitext(file_name)[0]}.txt")
            
            # Write the extracted data to the output file
            with open(output_file_path, 'w', encoding='utf-8') as output_file:
                for entry in extracted_data:
                    for heading, content in entry.items():
                        print(f"Heading: {heading}\nContent: {content}\n")
                        output_file.write(f"Heading: {heading}\nContent: {content}\n\n")
            
            print(f"Processed and saved: {output_file_path}")