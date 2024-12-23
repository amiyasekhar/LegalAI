# XML Contract Content Extractor

This script extracts content and paragraph properties from XML files (specifically those structured from Word documents) and saves the parsed information to a new text file. It’s designed to handle complex document structures, identifying paragraph properties and structuring paragraphs into coherent contract parts.

## Overview

The script performs the following steps for each XML file in a specified directory:
1. Parses the XML structure and identifies each paragraph along with its properties.
2. Extracts the document title from the first paragraph and treats subsequent paragraphs as parts of the contract.
3. Checks paragraph properties such as alignment, indentation, borders, shading, and spacing, saving these details for each paragraph.
4. Stops processing if a stop word (e.g., "END" or "STOP") is encountered.
5. Writes the extracted paragraphs and properties to a `.txt` file with the same name as the XML file, prefixed with `test2-`.

## Requirements

- Python 3.x

## Code Breakdown

### Key Components

- **Stop Words**: The script stops processing when it encounters a paragraph containing any word in the `stop_words` list (e.g., `['END', 'STOP']`).

- **Paragraph Properties**: The script uses a `para_properties` dictionary to extract various paragraph properties such as alignment, borders, shading, and spacing.

- **Namespaces**: To handle Word document-specific XML tags, several namespaces are defined.

### Functions

1. **is_bold**: Checks if a text run is bold.
2. **is_uppercase**: Checks if a paragraph is in uppercase.
3. **is_stop_word**: Determines if a paragraph contains a stop word.
4. **is_heading**: Checks if a paragraph text starts with "SECTION," indicating it’s likely a heading.
5. **get_para_text**: Extracts text from each paragraph, marking empty paragraphs with a placeholder.
6. **parse_paragraph_properties**: Extracts detailed properties for each paragraph, such as alignment, indentation, spacing, borders, and text direction.

### Main Function: `extract_headings_and_content_from_xml`

- Parses each XML file, identifies headings, and extracts properties for each paragraph.
- Combines paragraphs that don’t end in a period, forming coherent sections within the contract.
- Stops processing upon encountering a stop word.

### Usage

1. **Set Directory Path**: Update the `xml_directory` variable to specify the directory containing the XML files you want to process.
2. **Run the Script**:
   - The script iterates over all `.xml` files in the directory, processes each file, and saves extracted paragraphs and properties to a new `.txt` file with a `test2-` prefix.

### Output

For each XML file (e.g., `contract1.xml`), the script generates a text file (e.g., `test2-contract1-written-contract.txt`) with the following format: