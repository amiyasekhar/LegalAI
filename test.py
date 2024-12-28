def extract_clauses_from_file(filepath, delimiter):
    """
    Reads the entire file, then splits the text by a specific delimiter.
    Returns a list of clauses, stripped of leading/trailing whitespace.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
    
    # Split on the delimiter:
    raw_clauses = text.split(delimiter)
    
    # Strip whitespace and discard any empty strings:
    clauses = [clause.strip() for clause in raw_clauses if clause.strip()]
    return clauses

if __name__ == "__main__":
    # Example usage:
    contract_file = "broken_down_contract.txt"
    delimiter_str = "-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-"
    
    clauses_list = extract_clauses_from_file(contract_file, delimiter_str)
    print(clauses_list)
    
    '''
    print("Extracted Clauses:")
    for i, clause in enumerate(clauses_list, start=1):
        print(f"\n--- Clause {i} ---\n")
        print(clause)
    '''