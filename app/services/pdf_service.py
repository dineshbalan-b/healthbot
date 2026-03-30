import PyPDF2
from fastapi import UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf_upload(file: UploadFile):
    """Reads PDF text entirely dynamically from memory, and splits it into chunks."""
    text = ""
    reader = PyPDF2.PdfReader(file.file)
    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += f"\n\n--- Page {i+1} ---\n\n" + page_text

    if not text.strip():
        raise ValueError("Could not extract any understandable text from the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    texts = splitter.split_text(text)
    chunks = [{"title": f"{file.filename}_chunk_{i}", "text": t} for i, t in enumerate(texts)]
    return chunks
