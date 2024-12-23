import xml.etree.ElementTree as ET
from copy import deepcopy
import os
from xml.etree.ElementTree import QName

# Define stop words for the demo
stop_words = ['END', 'STOP']

# Initialising paragraph properties
para_properties = {
    "alignment": None,  # <w:jc w:val="left" "right" "center" "both" "distribute" "end" "start"/>
    "indent": {
        "left": None,      # <w:ind w:left="value"/>
        "right": None,     # <w:ind w:right="value"/>
        "firstLine": None, # <w:ind w:firstLine="value"/>
        "hanging": None    # <w:ind w:hanging="value"/>
    },
    "spacing": {
        "before": None,   # <w:spacing w:before="value"/>
        "after": None,    # <w:spacing w:after="value"/>
        "line": None,     # <w:spacing w:line="value"/>
        "lineRule": None  # <w:spacing w:lineRule="auto" "exact" "atLeast"/>
    },
    "borders": {
        "top": {
            "val": None,  # <w:top w:val="single" "dotted" "dashed"/>
            "size": None, # <w:top w:sz="value"/>
            "color": None # <w:top w:color="hex_value"/>
        },
        "left": {
            "val": None,  # <w:left w:val="single" "dotted" "dashed"/>
            "size": None, # <w:left w:sz="value"/>
            "color": None # <w:left w:color="hex_value"/>
        },
        "bottom": {
            "val": None,  # <w:bottom w:val="single" "dotted" "dashed"/>
            "size": None, # <w:bottom w:sz="value"/>
            "color": None # <w:bottom w:color="hex_value"/>
        },
        "right": {
            "val": None,  # <w:right w:val="single" "dotted" "dashed"/>
            "size": None, # <w:right w:sz="value"/>
            "color": None # <w:right w:color="hex_value"/>
        }
    },
    "shading": {
        "val": None,   # <w:shd w:val="clear" "solid"/>
        "color": None, # <w:shd w:color="hex_value"/>
        "fill": None   # <w:shd w:fill="hex_value"/>
    },
    "numbering": {
        "ilvl": None,  # <w:ilvl w:val="value"/>
        "numId": None  # <w:numId w:val="value"/>
    },
    "textDirection": None,    # <w:bidi w:val="true" "false"/>
    "outlineLevel": None,     # <w:outlineLvl w:val="integer_value"/>
    "pageBreakBefore": None,  # <w:pageBreakBefore w:val="true" "false"/>
    "keepLinesTogether": None,# <w:keepLines w:val="true" "false"/>
    "keepWithNext": None,     # <w:keepNext w:val="true" "false"/>
    "widowControl": None,     # <w:widowControl w:val="true" "false"/>
    "paragraphStyle": None,   # <w:pStyle w:val="style_name"/>
    "frameProperties": {
        "width": None,        # <w:framePr w:width="value"/>
        "height": None,       # <w:framePr w:height="value"/>
        "hSpace": None,       # <w:framePr w:hSpace="value"/>
        "vSpace": None        # <w:framePr w:vSpace="value"/>
    },
    "suppressLineNumbers": None,   # <w:suppressLineNumbers w:val="true" "false"/>
    "textAlignment": None,         # <w:textAlignment w:val="auto" "baseline" "top" "center" "bottom"/>
    "suppressAutoHyphens": None,   # <w:suppressAutoHyphens w:val="true" "false"/>
    "contextualSpacing": None,     # <w:contextualSpacing w:val="true" "false"/>
    "divId": None,                 # <w:divId w:val="value"/>
    "mirrorIndents": None,         # <w:mirrorIndents w:val="true" "false"/>
    "textDirectionVertical": None, # <w:textDirection w:val="lrTb" "tbRl"/>
    "dropCap": None,               # <w:dropCap w:val="none" "drop" "margin"/>
    "tabs": {
        "position": None,          # <w:tab w:pos="value"/>
        "val": None,               # <w:tab w:val="left" "right" "center" "decimal"/>
        "leader": None             # <w:tab w:leader="dots" "hyphen"/>
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
    return tag.split('}', 1)[1] if '}' in tag else tag

# Helper functions
def is_bold(run):
    global namespaces
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

def get_para_text(paragraph_elem, paragraph_properties):
    global namespaces
    paragraph_text = ""
    
    for run_elem in paragraph_elem.findall('.//w:r', namespaces=namespaces):
        text = ""
        run_text_elem = run_elem.find('.//w:t', namespaces=namespaces)
        if run_text_elem is not None and run_text_elem.text:
            text = run_text_elem.text
        else: # Empty text tag or no text tag, therefore we have an empty paragraph
            text = "-|-|-|**EMPTY PARA**|-|-|-"
        paragraph_text += text

    return paragraph_text

def parse_paragraph_properties(para_element):
    # Reset para_properties to ensure it's empty before filling
    global para_properties
    para_properties = deepcopy(para_properties)

    # Find the paragraph properties element
    pPr = para_element.find('w:pPr', namespaces)
    if pPr is not None:
        # Alignment
        jc = pPr.find('w:jc', namespaces)
        if jc is not None:
            para_properties['alignment'] = jc.get('{%s}val' % namespaces['w'])
        
        # Indentation
        ind = pPr.find('w:ind', namespaces)
        if ind is not None:
            para_properties['indent']['left'] = ind.get('{%s}left' % namespaces['w'])
            para_properties['indent']['right'] = ind.get('{%s}right' % namespaces['w'])
            para_properties['indent']['firstLine'] = ind.get('{%s}firstLine' % namespaces['w'])
            para_properties['indent']['hanging'] = ind.get('{%s}hanging' % namespaces['w'])
        
        # Spacing
        spacing = pPr.find('w:spacing', namespaces)
        if spacing is not None:
            para_properties['spacing']['before'] = spacing.get('{%s}before' % namespaces['w'])
            para_properties['spacing']['after'] = spacing.get('{%s}after' % namespaces['w'])
            para_properties['spacing']['line'] = spacing.get('{%s}line' % namespaces['w'])
            para_properties['spacing']['lineRule'] = spacing.get('{%s}lineRule' % namespaces['w'])
        
        # Borders
        pBdr = pPr.find('w:pBdr', namespaces)
        if pBdr is not None:
            for side in ['top', 'left', 'bottom', 'right']:
                border = pBdr.find('w:' + side, namespaces)
                if border is not None:
                    para_properties['borders'][side]['val'] = border.get('{%s}val' % namespaces['w'])
                    para_properties['borders'][side]['size'] = border.get('{%s}sz' % namespaces['w'])
                    para_properties['borders'][side]['color'] = border.get('{%s}color' % namespaces['w'])
        
        # Shading
        shd = pPr.find('w:shd', namespaces)
        if shd is not None:
            para_properties['shading']['val'] = shd.get('{%s}val' % namespaces['w'])
            para_properties['shading']['color'] = shd.get('{%s}color' % namespaces['w'])
            para_properties['shading']['fill'] = shd.get('{%s}fill' % namespaces['w'])
        
        # Numbering
        numPr = pPr.find('w:numPr', namespaces)
        if numPr is not None:
            ilvl = numPr.find('w:ilvl', namespaces)
            if ilvl is not None:
                para_properties['numbering']['ilvl'] = ilvl.get('{%s}val' % namespaces['w'])
            numId = numPr.find('w:numId', namespaces)
            if numId is not None:
                para_properties['numbering']['numId'] = numId.get('{%s}val' % namespaces['w'])
        
        # Text Direction
        bidi = pPr.find('w:bidi', namespaces)
        if bidi is not None:
            para_properties['textDirection'] = 'true'  # Presence of <w:bidi/> implies true
        
        # Outline Level
        outlineLvl = pPr.find('w:outlineLvl', namespaces)
        if outlineLvl is not None:
            para_properties['outlineLevel'] = outlineLvl.get('{%s}val' % namespaces['w'])
        
        # Page Break Before
        pageBreakBefore = pPr.find('w:pageBreakBefore', namespaces)
        if pageBreakBefore is not None:
            para_properties['pageBreakBefore'] = 'true'  # Presence implies true
        
        # Keep Lines Together
        keepLines = pPr.find('w:keepLines', namespaces)
        if keepLines is not None:
            para_properties['keepLinesTogether'] = 'true'
        
        # Keep With Next
        keepNext = pPr.find('w:keepNext', namespaces)
        if keepNext is not None:
            para_properties['keepWithNext'] = 'true'
        
        # Widow Control
        widowControl = pPr.find('w:widowControl', namespaces)
        if widowControl is not None:
            para_properties['widowControl'] = 'true'
        
        # Paragraph Style
        pStyle = pPr.find('w:pStyle', namespaces)
        if pStyle is not None:
            para_properties['paragraphStyle'] = pStyle.get('{%s}val' % namespaces['w'])
        
        # Frame Properties
        framePr = pPr.find('w:framePr', namespaces)
        if framePr is not None:
            para_properties['frameProperties']['width'] = framePr.get('{%s}width' % namespaces['w'])
            para_properties['frameProperties']['height'] = framePr.get('{%s}height' % namespaces['w'])
            para_properties['frameProperties']['hSpace'] = framePr.get('{%s}hSpace' % namespaces['w'])
            para_properties['frameProperties']['vSpace'] = framePr.get('{%s}vSpace' % namespaces['w'])
        
        # Suppress Line Numbers
        suppressLineNumbers = pPr.find('w:suppressLineNumbers', namespaces)
        if suppressLineNumbers is not None:
            para_properties['suppressLineNumbers'] = 'true'
        
        # Text Alignment
        textAlignment = pPr.find('w:textAlignment', namespaces)
        if textAlignment is not None:
            para_properties['textAlignment'] = textAlignment.get('{%s}val' % namespaces['w'])
        
        # Suppress Auto Hyphens
        suppressAutoHyphens = pPr.find('w:suppressAutoHyphens', namespaces)
        if suppressAutoHyphens is not None:
            para_properties['suppressAutoHyphens'] = 'true'
        
        # Contextual Spacing
        contextualSpacing = pPr.find('w:contextualSpacing', namespaces)
        if contextualSpacing is not None:
            para_properties['contextualSpacing'] = 'true'
        
        # Div ID
        divId = pPr.find('w:divId', namespaces)
        if divId is not None:
            para_properties['divId'] = divId.get('{%s}val' % namespaces['w'])
        
        # Mirror Indents
        mirrorIndents = pPr.find('w:mirrorIndents', namespaces)
        if mirrorIndents is not None:
            para_properties['mirrorIndents'] = 'true'
        
        # Text Direction Vertical
        textDirection = pPr.find('w:textDirection', namespaces)
        if textDirection is not None:
            para_properties['textDirectionVertical'] = textDirection.get('{%s}val' % namespaces['w'])
        
        # Drop Cap
        dropCap = pPr.find('w:dropCap', namespaces)
        if dropCap is not None:
            para_properties['dropCap'] = dropCap.get('{%s}val' % namespaces['w'])
        
        # Tabs
        tabs = pPr.find('w:tabs', namespaces)
        if tabs is not None:
            tab = tabs.find('w:tab', namespaces)
            if tab is not None:
                para_properties['tabs']['position'] = tab.get('{%s}pos' % namespaces['w'])
                para_properties['tabs']['val'] = tab.get('{%s}val' % namespaces['w'])
                para_properties['tabs']['leader'] = tab.get('{%s}leader' % namespaces['w'])
    
    return para_properties

# Function to extract headings and content from the XML contract
def extract_headings_and_content_from_xml(xml_file):
    global namespaces
    # Parse the XML
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Variables to store current state
    contract_title = ""
    is_title = False
    contract_parts = []
    stop_extraction = False
    checked_paragraphs = set()  # To keep track of processed paragraphs

    # Get the list of all paragraphs once
    paragraph_list = root.findall('.//w:p', namespaces=namespaces)

    # Iterate through paragraphs
    for paragraph_elem in paragraph_list:
        if stop_extraction:
            break  # Stop extraction if a stop word is encountered

        if paragraph_elem in checked_paragraphs:
            continue  # Skip paragraphs that have already been processed

        # Check if this is the title (first paragraph)
        if paragraph_elem == paragraph_list[0]:
            is_title = True

        para_properties = parse_paragraph_properties(paragraph_elem)

        # Collect all text in the paragraph
        paragraph_text = get_para_text(paragraph_elem, para_properties)
        checked_paragraphs.add(paragraph_elem)

        # If paragraph is title, set the title
        if is_title:
            contract_title = paragraph_text
            contract_parts.append(contract_title)
            is_title = False  # Reset the flag after setting the title
            continue

        else:
            if paragraph_text != "":
                contract_parts.append(paragraph_text)
            '''
            # If the paragraph doesn't end with a period, combine it with subsequent paragraphs
            if not paragraph_text.rstrip().endswith('.'):
                idx = paragraph_list.index(paragraph_elem)
                # Start from the next paragraph
                for i in range(idx + 1, len(paragraph_list)):
                    next_paragraph_elem = paragraph_list[i]

                    # Skip if already checked
                    if next_paragraph_elem in checked_paragraphs:
                        continue

                    next_para_properties = parse_paragraph_properties(next_paragraph_elem)
                    next_paragraph_text = get_para_text(next_paragraph_elem, next_para_properties)

                    # Append the text of the next paragraph
                    paragraph_text += ' ' + next_paragraph_text

                    # Mark the next paragraph as checked
                    checked_paragraphs.add(next_paragraph_elem)

                    # Check if the next paragraph ends with a period
                    if next_paragraph_text.rstrip().endswith('.'):
                        break  # Stop combining paragraphs
                    else:
                        # Continue to the next paragraph
                        continue
            '''
            

    return contract_parts

# Main function to test the extraction
if __name__ == "__main__":
    # Directory containing the XML files
    xml_directory = '/Users/amiyasekhar/CLM/contracts'

    # List all files in the directory
    files = os.listdir(xml_directory)

    # Process each .xml file
    for file_name in files:
        if file_name.lower().endswith('.xml'):
            xml_file_path = os.path.join(xml_directory, file_name)
            # Remove the .xml extension to get the base name
            base_name = os.path.splitext(file_name)[0]
            output_txt_name = f"{base_name}-written-contract.txt"
            output_txt_path = os.path.join("/Users/amiyasekhar/CLM/beautified_contract_to_txt", output_txt_name)
            try:
                # Extract headings and content
                extracted_data = extract_headings_and_content_from_xml(xml_file_path)

                # Write the extracted data to the output file
                with open(output_txt_path, 'w', encoding='utf-8') as written_contract:
                    for entry in extracted_data:
                        written_contract.write(f"Entry:  {entry}\n")
                print(f"Written contract saved to {output_txt_path}")
            except Exception as e:
                print(f"Error processing {xml_file_path}: {e}")