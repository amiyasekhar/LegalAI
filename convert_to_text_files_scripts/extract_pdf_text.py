import fitz  # PyMuPDF

def pdf_to_formatted_txt_with_right_spacing_and_string(pdf_path, output_txt_path):
    doc = fitz.open(pdf_path)
    output_content = ""  # String to store everything written to the txt file

    with open(output_txt_path, "w") as txt_file:
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            blocks = page.get_text("dict")["blocks"]
            page_width = page.rect.width
            
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        line_text = ""  # We'll construct line text incrementally
                        last_x0 = 0  # Starting x of the last span
                        last_x1 = 0  # Ending x of the last span
                        
                        for span in line["spans"]:
                            text = span["text"].strip()
                            bbox = span["bbox"]
                            x0, x1 = bbox[0], bbox[2]  # Current span's starting and ending x coordinates

                            # Determine the spacing based on the difference between the current x0 and the last x1
                            if last_x1 == 0:  # This is the first span in the line
                                initial_spaces = int(x0 // 5)  # Add spaces at the beginning of the line
                                line_text += " " * initial_spaces
                            else:
                                # Add spaces between spans based on the gap
                                gap = int((x0 - last_x1) // 5)
                                line_text += " " * gap
                            
                            line_text += text
                            last_x0, last_x1 = x0, x1  # Update last_x0 and last_x1 for the next span

                        # After processing the line, calculate right-side spacing
                        if last_x1 < page_width:
                            right_spaces = int((page_width - last_x1) // 5)
                            line_text += " " * right_spaces

                        # Print the constructed line text
                        print(f"Line: {line_text if line_text.strip() else '[Empty Line]'}")

                        # Write the constructed line text to file and also append to the string
                        txt_file.write(line_text + "\n")
                        output_content += line_text + "\n"

                    # Add extra newline for separation between blocks
                    txt_file.write("\n")
                    output_content += "\n"

    print(f"Formatted text with right-side spacing has been written to {output_txt_path}.")
    
    return output_content  # Return the entire content as a string

# Example usage
pdf_path = '/Users/amiyasekhar/CLM/contracts/5886-united-states-and-japan-transition-mineral-agreement-2023.pdf'  # Replace with your actual PDF file
output_txt_path = 'PDF_TO_TXT_5886-united-states-and-japan-transition-mineral-agreement-2023.txt'

output_content = pdf_to_formatted_txt_with_right_spacing_and_string(pdf_path, output_txt_path)
print(f"PDF content stored in string format:\n{output_content}")