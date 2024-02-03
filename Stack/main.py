import fitz
import re

# Open the input PDF file
inputFileName = "Bangla Class 9 RB.pdf"
doc = fitz.open(inputFileName)

# Define a new ToUnicode stream content
new = '1 beginbfrange\n<c0> <ff> <0410>\nendbfrange'

# Iterate through each page in the PDF document
for pno in range(doc.page_count):
    # Get the font tuples used on the current page
    font_tuples = doc.get_page_fonts(pno)

    # Iterate through each font tuple
    for font_tuple in font_tuples:
        # Get the ToUnicode stream content for the current font
        for line in doc.xref_object(font_tuple[0]).splitlines():
            line = line.strip()
            
            # Find the line containing "/ToUnicode"
            if line.startswith("/ToUnicode"):
                # Extract the stream ID from the line
                stream_id = int(line.split()[1])
                
                # Get the existing ToUnicode stream content and decode it
                old_stream_decoded = doc.xref_stream(stream_id).decode()
                
                # Replace the existing ToUnicode stream content with the new content
                new_stream_decoded = re.sub('[0-9]+? beginbfrange.*endbfrange', new, old_stream_decoded, flags=re.DOTALL)
                
                # Encode the new ToUnicode stream content
                new_stream_encoded = new_stream_decoded.encode()
                
                # Update the ToUnicode stream in the PDF document
                doc.update_stream(stream_id, new_stream_encoded)

# Save the modified PDF to an output file
outputFileName = "your_output_file.pdf"
doc.save(outputFileName)
