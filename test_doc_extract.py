import pypandoc

# Example file:
docxFilename = '/Users/amiyasekhar/Downloads/CLM CW/contracts/1.Contract.docx'
output = pypandoc.convert_file(docxFilename, 'plain', outputfile="somefile.txt")
assert output == ""

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python docx_to_txt.py input.docx output.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    docx_to_txt(input_file, output_file)
