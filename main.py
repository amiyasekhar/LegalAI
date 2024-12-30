from extract_clauses import (
    docx_to_formatted_txt_with_right_spacing,
    split_into_chunks,
    write_chunks_to_files,
    generate_constituent_parts,
    debug_log
)

from cc_for_real import main as classify_clauses

import os

def process_contract(contract_path, output_dir="./output"):
    """
    Process a contract document through extraction and classification
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Convert DOCX to text if needed
        debug_log("Starting document conversion...")
        if contract_path.endswith('.docx'):
            contract_text = docx_to_formatted_txt_with_right_spacing(contract_path)
            txt_path = contract_path[:-5] + ".txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(contract_text)
            contract_path = txt_path
        
        # Step 2: Split into chunks
        debug_log("Splitting document into chunks...")
        chunks = split_into_chunks(
            file_path=contract_path,
            chunk_size=128000,
            model="gpt-4"
        )
        
        # Step 3: Write chunks to files
        chunks_dir = os.path.join(output_dir, "chunks")
        debug_log("Writing chunks to files...")
        write_chunks_to_files(chunks, chunks_dir)
        
        # Step 4: Extract clauses from each chunk
        debug_log("Extracting clauses...")
        extracted_clauses = []
        for filename in sorted(os.listdir(chunks_dir)):
            if filename.startswith('chunk_'):
                file_path = os.path.join(chunks_dir, filename)
                with open(file_path, "r", encoding="utf-8") as chunk_file:
                    chunk_content = chunk_file.read()
                    result = generate_constituent_parts(chunk_content)
                    extracted_clauses.append(result)
        
        # Write extracted clauses to a single file
        broken_down_path = os.path.join(output_dir, "broken_down_contract.txt")
        with open(broken_down_path, "w", encoding="utf-8") as out_file:
            for clauses in extracted_clauses:
                out_file.write(clauses + "\n")
        
        # Step 5: Run classification on the extracted clauses
        debug_log("Starting classification...")
        classify_clauses()
        
        debug_log("Processing complete!")
        return True
        
    except Exception as e:
        debug_log(f"Error in process_contract: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    contract_path = "/path/to/your/contract.docx"  # Replace with your contract path
    output_dir = "./processed_contracts"            # Replace with your desired output directory
    
    try:
        process_contract(contract_path, output_dir)
        print("Contract processing completed successfully!")
    except Exception as e:
        print(f"Error processing contract: {str(e)}")