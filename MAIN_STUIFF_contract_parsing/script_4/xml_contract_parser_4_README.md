# XML Text Extractor with Formatting for Word Documents

This script processes XML files, particularly those extracted from Word documents (`.docx`), and extracts the text content with its associated formatting (e.g., bold, italic, underline). The output includes list structures, table content, and paragraphs with Markdown-compatible syntax, saved as text files.

## Overview

The script performs the following steps for each XML file in a specified directory:
1. Parses the XML structure, focusing on `<w:body>` to retrieve the document’s main content.
2. Extracts text and applies formatting (e.g., bold, italic, color, list structure) to preserve the visual structure.
3. Handles list elements, tables, and paragraphs, formatting them for readability in the output.
4. Writes the extracted and formatted content into `.txt` files with Markdown syntax to replicate document styling.

## Requirements

- Python 3.x

## Code Breakdown

### Key Components

- **Namespaces**: The script defines namespaces specific to Word document XML structures for easy parsing.
- **Formatting Extraction**: Extracts text styling attributes (bold, italic, underline, color, etc.) from XML tags and applies them using Markdown or HTML syntax in the output.
- **List and Table Support**: Recognizes lists and tables, formats them accordingly, and includes indentation for list levels.
- **Debug Output**: Writes debug information for each processed XML file to a `debug.txt` file to assist in troubleshooting.

### Functions

1. **strip_namespace**: Removes namespaces from XML tags for simpler handling.
2. **parse_xml**: Parses the XML file and returns the root element.
3. **get_text_with_formatting**: Recursively retrieves text content from XML elements, applying the appropriate formatting.
4. **apply_formatting**: Applies Markdown and HTML formatting based on extracted text properties.
5. **get_list_info**: Determines if a paragraph belongs to a list and identifies its level and type.
6. **get_list_marker**: Returns the appropriate list marker (bullet or number) based on list type and level.
7. **extract_text_from_xml**: Main function that extracts formatted text and writes it to a debug file and output file.

### Usage

1. **Set Directory Paths**: Update `xml_directory` and the output directory to specify the XML input and text output directories.
2. **Run the Script**:
   - The script iterates over all `.xml` files in the specified directory, processes each file, and saves extracted text with Markdown-compatible formatting in a new `.txt` file.

### Output

For each XML file (e.g., `contract1.xml`), the script generates a text file (e.g., `contract1-written-contract-md-encoded.txt`) in the output directory. The output uses Markdown syntax where possible for formatting, and includes tables and lists, as in the original document. An example output: