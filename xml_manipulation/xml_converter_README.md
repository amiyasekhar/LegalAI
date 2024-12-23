# DOCX to Beautified XML Extractor

This script extracts and beautifies the XML content of each `.docx` file in a specified directory, saving the structured XML to a readable `.xml` file. It allows you to inspect or process the internal structure of Word documents in XML format.

## Overview

The script performs the following steps:
1. Opens each `.docx` file as a ZIP archive.
2. Extracts the main XML content from `document.xml`.
3. Beautifies the XML content by adding indentation and line breaks for readability.
4. Saves the beautified XML as a `.xml` file in the same directory as the `.docx`.

## Requirements

- Python 3.x

## Code Breakdown

### Imports

- `zipfile`: Opens the `.docx` file as a ZIP archive.
- `xml.dom.minidom`: Parses and beautifies XML content.
- `os`: Provides functions for filesystem operations.

### Function: `extract_and_beautify_xml_from_docx`

This function handles the extraction and beautification of XML content from a `.docx` file.

```python
def extract_and_beautify_xml_from_docx(docx_path, output_xml_path):
    # Open the .docx file as a zip
    with zipfile.ZipFile(docx_path, 'r') as docx_zip:
        # Locate and read the document.xml file from the docx archive
        xml_content = docx_zip.read('word/document.xml')
        
        # Parse the XML content for beautification
        dom = xml.dom.minidom.parseString(xml_content)
        beautified_xml = dom.toprettyxml(indent="  ")

        # Write the beautified XML content to a new file
        with open(output_xml_path, 'w', encoding='utf-8') as xml_file:
            xml_file.write(beautified_xml)
