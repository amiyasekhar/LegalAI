import pandas as pd

excel_file = "/Users/amiyasekhar/CLM/clause_content_variety_latest_clauses.xlsx"

df = pd.read_excel(excel_file)

# Define replacements
replacements = {
    "intellectual property rights": "intellectual property",
    "notice": "notice"
}

# Apply replacements to the "Clause Heading" column
df["Clause Heading"] = df["Clause Heading"].replace(replacements)

# Overwrite the original Excel file (no new file created)
df.to_excel(excel_file, index=False)

print(f"Done! '{excel_file}' overwritten with updated clause headings.")