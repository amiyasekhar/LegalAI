import xml.etree.ElementTree as ET
import os

# Define the namespaces used in the WordprocessingML document
namespaces = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'm': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
    # Add any other namespaces as needed
}

def strip_namespace(tag):
    """Strip the namespace from the XML tag."""
    return tag.split('}')[-1]

def parse_xml(xml_file):
    """Parse the XML file and get the root element."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    return root

def get_text_with_formatting(element, inherited_formatting=None, list_level=0):
    """
    Recursively extract text from the element, along with formatting.
    """
    texts = []
    if inherited_formatting is None:
        inherited_formatting = {}

    # Check if the element is a run (<w:r>)
    if strip_namespace(element.tag) == 'r':
        # Get formatting from <w:rPr>
        run_formatting = inherited_formatting.copy()
        rPr = element.find('w:rPr', namespaces)
        if rPr is not None:
            for prop in rPr:
                prop_tag = strip_namespace(prop.tag)
                if prop_tag == 'b':
                    run_formatting['bold'] = True
                elif prop_tag == 'i':
                    run_formatting['italic'] = True
                elif prop_tag == 'u':
                    run_formatting['underline'] = prop.get('{%s}val' % namespaces['w'], 'single')
                elif prop_tag == 'strike':
                    run_formatting['strikethrough'] = True
                elif prop_tag == 'color':
                    run_formatting['color'] = prop.get('{%s}val' % namespaces['w'])
                elif prop_tag == 'sz':
                    run_formatting['size'] = int(prop.get('{%s}val' % namespaces['w']))
                elif prop_tag == 'highlight':
                    run_formatting['highlight'] = prop.get('{%s}val' % namespaces['w'])
                # Add more formatting properties as needed

        # Get the text content from <w:t> elements
        for t in element.findall('w:t', namespaces):
            text = t.text
            if text:
                formatted_text = apply_formatting(text, run_formatting)
                texts.append(formatted_text)

        # Handle other possible child elements within <w:r>, like <w:tab>, <w:br>, etc.
        for child in element:
            child_tag = strip_namespace(child.tag)
            if child_tag == 'tab':
                texts.append('\t')  # Represent tab as actual tab character or spaces
            elif child_tag == 'br':
                texts.append('\n')  # Line break
            elif child_tag == 'cr':
                texts.append('\n')  # Carriage return
            # Handle other run-level elements as needed

    elif strip_namespace(element.tag) == 'p':
        # It's a paragraph
        paragraph_formatting = inherited_formatting.copy()
        list_info = get_list_info(element)
        if list_info:
            list_level = list_info['level']
            paragraph_formatting['list_type'] = list_info['type']
        else:
            paragraph_formatting.pop('list_type', None)
            list_level = 0

        # Process runs within the paragraph
        for child in element:
            child_tag = strip_namespace(child.tag)
            if child_tag == 'r':
                texts.extend(get_text_with_formatting(child, paragraph_formatting, list_level))
            elif child_tag == 'pPr':
                # Get paragraph-level formatting, such as alignment
                pPr = child
                for prop in pPr:
                    prop_tag = strip_namespace(prop.tag)
                    if prop_tag == 'jc':
                        paragraph_formatting['alignment'] = prop.get('{%s}val' % namespaces['w'])
                    # Handle other paragraph properties as needed
            elif child_tag == 'hyperlink':
                # Process hyperlink content
                texts.extend(get_text_with_formatting(child, paragraph_formatting, list_level))
            elif child_tag in ['bookmarkStart', 'bookmarkEnd']:
                # Ignore bookmarks for text content
                pass
            elif child_tag == 'fldSimple':
                # Handle simple field (e.g., page number)
                texts.extend(get_text_with_formatting(child, paragraph_formatting, list_level))
            # Handle other paragraph-level elements as needed

        # Handle list formatting
        if 'list_type' in paragraph_formatting:
            # Prepend list marker to each text entry
            list_marker = get_list_marker(paragraph_formatting['list_type'], list_level)
            formatted_texts = []
            for text in texts:
                if text.strip() != '':
                    formatted_texts.append(' ' * (list_level * 2) + list_marker + text)
            texts = formatted_texts

        # Add line break at the end of the paragraph
        texts.append('\n')

    elif strip_namespace(element.tag) == 'tbl':
        # It's a table
        table_text = []
        for row in element.findall('w:tr', namespaces):
            row_text = []
            for cell in row.findall('w:tc', namespaces):
                cell_texts = []
                for cell_child in cell:
                    cell_child_tag = strip_namespace(cell_child.tag)
                    if cell_child_tag in ['p', 'tbl']:
                        cell_texts.extend(get_text_with_formatting(cell_child, inherited_formatting))
                # Combine cell texts and represent them in a table-like structure
                cell_text = ''.join(cell_texts).strip()
                row_text.append(f"| {cell_text} ")
            # Add end of table row
            row_text.append('|')
            table_text.append(''.join(row_text) + '\n')
        # Combine all rows
        texts.extend(table_text)
        # Add separation line after the table
        texts.append('\n')

    else:
        # Process other elements recursively
        for child in element:
            texts.extend(get_text_with_formatting(child, inherited_formatting, list_level))

    return texts

def apply_formatting(text, formatting):
    """Apply formatting to the text using Markdown syntax."""
    if formatting.get('bold'):
        text = f'**{text}**'
    if formatting.get('italic'):
        text = f'*{text}*'
    if formatting.get('underline'):
        # Markdown does not support underline, use HTML or custom syntax
        text = f'<u>{text}</u>'
    if formatting.get('strikethrough'):
        text = f'~~{text}~~'
    # Handle font size, color, etc., as needed (using HTML spans or other methods)
    if formatting.get('color'):
        color = formatting['color']
        text = f'<span style="color:#{color}">{text}</span>'
    if formatting.get('highlight'):
        highlight = formatting['highlight']
        text = f'<mark>{text}</mark>'
    # Handle other formatting properties as needed
    return text

def get_list_info(paragraph):
    """Determine if the paragraph is part of a list and return list info."""
    pPr = paragraph.find('w:pPr', namespaces)
    if pPr is not None:
        numPr = pPr.find('w:numPr', namespaces)
        if numPr is not None:
            ilvl_elem = numPr.find('w:ilvl', namespaces)
            numId_elem = numPr.find('w:numId', namespaces)
            if ilvl_elem is not None and numId_elem is not None:
                level = int(ilvl_elem.get('{%s}val' % namespaces['w']))
                numId = numId_elem.get('{%s}val' % namespaces['w'])
                # For simplicity, we can assume numId corresponds to bullet or number
                # In practice, you'd need to look up the numbering format in numbering.xml
                list_type = get_list_type(numId)
                return {'level': level, 'type': list_type}
    return None

def get_list_type(numId):
    """Determine the list type based on numId."""
    # This function would need to parse numbering.xml to get the actual list format
    # For simplicity, let's assume numId '1' is a numbered list, others are bullets
    if numId == '1':
        return 'numbered'
    else:
        return 'bulleted'

def get_list_marker(list_type, level):
    """Get the list marker based on list type and level."""
    if list_type == 'numbered':
        return '1. '  # Simplification; numbering should increment
    else:
        return '- '  # Bullet point

def extract_text_from_xml(xml_file):
    """Main function to extract text with formatting from the XML file."""
    root = parse_xml(xml_file)
    body = root.find('w:body', namespaces)
    texts = []
    index = 1
    with open('./xml_contract_parser_4_debug.txt', 'a') as db:
        db.write(xml_file + "\n")
    for child in body:
        child_texts = get_text_with_formatting(child)
        for text in child_texts:
            texts.append({f"Child {index}": child_texts})
        with open('./debug.txt', 'a') as db:
            db.write(f"Child {index}: {child_texts}\n")
        index += 1  # Increment index for each child
    with open('./debug.txt', 'a') as db:
        db.write("\n")
        
    return texts

# Example usage
if __name__ == '__main__':
    # Directory containing the XML files
    xml_directory = '/Users/amiyasekhar/CLM/contracts'

    # List all files in the directory
    try:
        files = os.listdir(xml_directory)
    except FileNotFoundError:
        print(f"The directory {xml_directory} does not exist.")
        exit(1)

    # Process each .xml file
    for file_name in files:
        if file_name.lower().endswith('.xml'):
            xml_file_path = os.path.join(xml_directory, file_name)
            # Remove the .xml extension to get the base name
            base_name = os.path.splitext(file_name)[0]
            output_txt_name = f"{base_name}-written-contract-md-encoded.txt"
            output_txt_path = os.path.join("/Users/amiyasekhar/CLM/MAIN_STUIFF_contract_parsing/results_4", output_txt_name)
            try:
                # Extract headings and content
                extracted_texts = extract_text_from_xml(xml_file_path)

                # Write the extracted data to the output file
                with open(output_txt_path, 'w', encoding='utf-8') as written_contract:
                    for child in extracted_texts:
                        for child_key, child_texts in child.items():
                            written_contract.write(f"{child_key}:\n")
                            for entry in child_texts:
                                written_contract.write(f"Entry: {entry}\n")
                            written_contract.write("\n")
                print(f"Written contract saved to {output_txt_path}")
            except ET.ParseError as pe:
                print(f"XML parsing error in {xml_file_path}: {pe}")
            except Exception as e:
                print(f"Error processing {xml_file_path}: {e}")