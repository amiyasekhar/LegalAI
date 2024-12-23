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
    return tag.split('}', 1)[-1] if '}' in tag else tag

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
                # Get paragraph-level formatting, such as alignment and style
                pPr = child
                for prop in pPr:
                    prop_tag = strip_namespace(prop.tag)
                    if prop_tag == 'jc':
                        paragraph_formatting['alignment'] = prop.get('{%s}val' % namespaces['w'])
                    elif prop_tag == 'rStyle':
                        paragraph_formatting['style'] = prop.get('{%s}val' % namespaces['w'])
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
                    indentation = '  ' * list_level  # Two spaces per level
                    formatted_texts.append(f"{indentation}{list_marker} {text}")
            texts = formatted_texts

        # Handle headings based on font size or style
        heading_level = get_heading_level(paragraph_formatting)
        if heading_level:
            # Prepend heading markdown
            texts = [f"{'#' * heading_level} {text}" for text in texts]

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
                row_text.append(f"{cell_text}")
            # Join cell texts with pipe '|' to mimic table structure
            table_row = '| ' + ' | '.join(row_text) + ' |'
            table_text.append(table_row)
        # Add separation line after the header row if the first row is a header
        if table_text:
            separator = '| ' + ' | '.join(['---'] * (len(table_text[0].split('|')) - 2)) + ' |'
            table_text.insert(1, separator)
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
        # Markdown does not support underline; using HTML syntax
        text = f'<u>{text}</u>'
    if formatting.get('strikethrough'):
        text = f'~~{text}~~'
    # Handle font size and color using HTML spans for more control
    if formatting.get('size'):
        size = formatting['size']
        # Define a mapping from font size to heading levels
        heading_level = map_font_size_to_heading(size)
        if heading_level:
            text = f"{'#' * heading_level} {text}"
    if formatting.get('color'):
        color = formatting['color']
        text = f'<span style="color:#{color}">{text}</span>'
    if formatting.get('highlight'):
        highlight = formatting['highlight']
        text = f'<mark>{text}</mark>'
    return text

def map_font_size_to_heading(font_size):
    """
    Map font size to Markdown heading level.
    For example:
    - 48 (24pt) -> Heading 1
    - 36 (18pt) -> Heading 2
    - 24 (12pt) -> Heading 3
    Adjust these mappings based on your specific font size usage.
    """
    size_to_heading = {
        48: 1,  # 24pt
        36: 2,  # 18pt
        24: 3,  # 12pt
        18: 4,  # 9pt
        16: 5,  # 8pt
        14: 6,  # 7pt
        # Add more mappings as needed
    }
    return size_to_heading.get(font_size, None)

def get_heading_level(paragraph_formatting):
    """Determine heading level based on font size or paragraph style."""
    # First, check if a style is applied (e.g., Heading1, Heading2)
    style = paragraph_formatting.get('style', '').lower()
    if 'heading1' in style:
        return 1
    elif 'heading2' in style:
        return 2
    elif 'heading3' in style:
        return 3
    elif 'heading4' in style:
        return 4
    elif 'heading5' in style:
        return 5
    elif 'heading6' in style:
        return 6
    # If no style, map based on font size
    elif 'size' in paragraph_formatting:
        return map_font_size_to_heading(paragraph_formatting['size'])
    else:
        return None  # Not a heading

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
                # For simplicity, we can assume numId '1' is a numbered list, others are bullets
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
        return '1.'  # Simplification; Markdown allows '1.' for all items
    else:
        return '-'  # Bullet point

def extract_text_from_xml(xml_file):
    """Main function to extract text with formatting from the XML file."""
    root = parse_xml(xml_file)
    body = root.find('w:body', namespaces)
    texts = []
    index = 1
    with open('./3.1_debug.txt', 'a', encoding='utf-8') as db:
        db.write(f"Processing file: {xml_file}\n")
    for child in body:
        child_texts = get_text_with_formatting(child)
        combined_text = ''.join(child_texts)
        
        # Handle <w:p> without <w:t> by checking if it's a paragraph and has no text
        if strip_namespace(child.tag) == 'p' and not combined_text.strip():
            combined_text = "\n"

        # Skip if combined_text is empty and not a newline
        if combined_text and not is_stop_word(combined_text):
            texts.append({f"Child {index}": combined_text})
            with open('./3.1_debug.txt', 'a', encoding='utf-8') as db:
                db.write(f"Child {index}: {combined_text}\n")
            index += 1  # Increment only when a child has text
        elif combined_text == "\n" and strip_namespace(child.tag) == 'p':
            # Represent empty paragraph as newline
            texts.append({f"Child {index}": combined_text})
            with open('./3.1_debug.txt', 'a', encoding='utf-8') as db:
                db.write(f"Child {index}: {combined_text}\n")
            index += 1

    with open('./3.1_debug.txt', 'a', encoding='utf-8') as db:
        db.write("\n")
    return texts

def process_xml_files(xml_directory):
    """
    Process all .xml files in the given directory, extract text, and save to .md files.
    
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
            output_md_name = f"{base_name}-written-contract.md"
            output_md_path = os.path.join("/Users/amiyasekhar/CLM/MAIN_STUIFF_contract_parsing/results_5", output_md_name)

            extracted_data = extract_text_from_xml(xml_file_path)

            if extracted_data:
                try:
                    with open(output_md_path, 'w', encoding='utf-8') as out_file:
                        for child in extracted_data:
                            for child_key, child_text in child.items():
                                out_file.write(f"{child_key}:\n{child_text}\n\n")
                    print(f"Markdown contract saved to '{output_md_path}'.")
                except Exception as e:
                    print(f"Error writing to '{output_md_path}': {e}")
            else:
                print(f"No relevant text extracted from '{xml_file_path}'.")

def is_stop_word(text):
    """Check if the text contains any stop word."""
    stop_words = ['END', 'STOP']
    return any(word in text.upper() for word in stop_words)

# Example usage
if __name__ == '__main__':
    # Directory containing the XML files
    xml_directory = '/Users/amiyasekhar/CLM/contracts'  # Update this path as needed

    # Clear the debug file at the start
    open('./xml_contract_parser_5_debug.txt', 'w').close()

    # Process XML files
    process_xml_files(xml_directory)