# XML to Markdown Document Converter

This script processes XML files, specifically those extracted from Word documents (`.docx`), and extracts their text content along with formatting information. The output is generated in Markdown format, preserving the document’s structure, lists, tables, headings, and text styling (e.g., bold, italic).

## Overview

The script performs the following steps for each XML file in a specified directory:
1. **Parse the XML**: Focuses on the `<w:body>` element containing the main content.
2. **Extract Text and Formatting**: Retrieves text from XML tags and applies formatting, including:
   - **Bold**, *Italic*, and <u>Underline</u> using Markdown and HTML syntax.
   - **Font Size Mapping**: Converts certain font sizes to Markdown headings.
   - **Color and Highlight**: Uses HTML `<span>` and `<mark>` tags for color and highlight effects.
3. **Handle Lists and Tables**: Detects lists and tables, preserving their structure.
4. **Save Output as Markdown**: Writes formatted text into `.md` files for easy readability.

## Requirements

- Python 3.x

## Code Breakdown

### Key Components

- **Namespaces**: Defines namespaces used in WordprocessingML documents for consistent XML parsing.
- **Formatting Extraction**: Extracts text styling attributes (bold, italic, underline, color, etc.) and applies Markdown or HTML styling.
- **List and Table Support**: Recognizes lists and tables, formatting them in Markdown syntax.
- **Debug Output**: Writes debug information for each processed XML file to `3.1_debug.txt` for troubleshooting.

### Functions

1. **strip_namespace**: Removes namespaces from XML tags for easier handling.
2. **parse_xml**: Parses the XML file and returns the root element.
3. **get_text_with_formatting**: Recursively retrieves text content from XML elements, applying the appropriate formatting.
4. **apply_formatting**: Applies Markdown and HTML formatting based on extracted text properties.
5. **map_font_size_to_heading**: Maps font sizes to Markdown heading levels.
6. **get_list_info**: Determines if a paragraph is part of a list and identifies list type and level.
7. **get_list_marker**: Returns the correct list marker (bullet or number) based on list type and level.
8. **extract_text_from_xml**: Main function that extracts formatted text and writes it to a debug file and Markdown output file.

### Usage

1. **Set Directory Path**: Update `xml_directory` to specify the directory containing the XML files to be processed.
2. **Run the Script**:
   - The script iterates over all `.xml` files in the specified directory, processes each file, and saves the formatted output to new `.md` files.

### Output

For each XML file (e.g., `contract1.xml`), the script generates a Markdown file (e.g., `contract1-written-contract.md`) with the following format:

```markdown
Child 1:
# Heading Text

Entry: This is the content of the first paragraph with **bold** and *italic* text.

Child 2:
1. List item 1
    - Sub-item 1.1
    - Sub-item 1.2

Child 3:
| Table Cell 1 | Table Cell 2 |
|--------------|--------------|
| Data 1       | Data 2       |
