import docx2txt
import glob

directory = glob.glob('/Users/amiyasekhar/CLM/contracts/1.Contract.docx')

for file_name in directory:
    with open(file_name, 'rb') as infile:
        with open(file_name[:-5]+'.txt', 'w', encoding='utf-8') as outfile:
            doc = docx2txt.process(infile)
            outfile.write(doc)

print("=========")
print("All done!")