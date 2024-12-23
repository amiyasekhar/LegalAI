import zipfile
import xml.dom.minidom
import os

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

# Usage

# Define the directory containing the .docx files
docx_directory = '/Users/amiyasekhar/CLM/contracts'

# List all files in the directory
files = os.listdir(docx_directory)

# Process each .docx file
for file_name in files:
    if file_name.lower().endswith('.docx'):
        docx_path = os.path.join(docx_directory, file_name)
        # Remove the .docx extension and add .xml
        base_name = os.path.splitext(file_name)[0]
        output_xml_name = f"{base_name}.xml"
        output_xml_path = os.path.join(docx_directory, output_xml_name)
        
        try:
            # Extract and beautify the XML content
            extract_and_beautify_xml_from_docx(docx_path, output_xml_path)
            print(f"Beautified XML content saved to {output_xml_path}")
        except Exception as e:
            print(f"Error processing {docx_path}: {e}")