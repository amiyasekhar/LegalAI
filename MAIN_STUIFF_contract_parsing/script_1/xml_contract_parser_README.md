# XML Contract Parser

This script processes XML files (specifically those structured from Word documents) to extract headings and their associated content. Each XML file is parsed to identify headings, stop words, and content sections, and the results are saved to separate text files for easy readability and inspection.

## Overview

The script performs the following steps for each XML file in a specified directory:
1. Parses the XML structure and identifies each paragraph.
2. Determines if the paragraph is a heading (e.g., starts with "SECTION" or is in uppercase) or regular content.
3. Stops processing further content in a file if a stop word (e.g., "END" or "STOP") is encountered.
4. Writes the extracted headings and content pairs to a `.txt` file with the same name as the original XML file.

## Requirements

- Python 3.x

## Code Breakdown

### Helper Functions

- **is_bold**: Checks if a text run is bold (though not currently used in heading determination).
- **is_uppercase**: Checks if a paragraph is in uppercase.
- **is_stop_word**: Checks if a paragraph contains a stop word to halt further extraction.
- **is_heading**: Determines if a paragraph qualifies as a heading (starts with "SECTION" or is uppercase).

### Main Function: `extract_headings_and_content_from_xml`

- Parses each XML file, identifies headings and content based on paragraph properties, and stops further extraction upon encountering a stop word.
- The function appends each heading-content pair to a list, which is written to an output `.txt` file.

### Usage

1. **Set Directory Path**: Update the `directory_path` variable to specify the directory containing the XML files you want to process.
2. **Run the Script**:
   - The script iterates over all `.xml` files in the directory, processes each file, and saves extracted headings and content to corresponding `.txt` files.

### Output

For each XML file (e.g., `contract1.xml`), the script generates a text file (e.g., `contract1.txt`) with the following format: