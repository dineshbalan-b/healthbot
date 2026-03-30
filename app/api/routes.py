from fastapi import APIRouter, UploadFile, File, HTTPException
from app.models.schemas import ChatRequest, ChatResponse
from app.services.pdf_service import process_pdf_upload
from app.services.rag_service import get_vector_store, add_documents_to_store, get_rag_chain

router = APIRouter()

@router.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
    
    try:
        chunks = process_pdf_upload(file)
        add_documents_to_store(chunks)
        return {"message": "PDF processed and vector database updated successfully. You can now ask questions!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    if not get_vector_store():
        raise HTTPException(status_code=400, detail="No knowledge base available. Please upload a PDF first.")

    input_text = request.question.strip()
    if not input_text:
        raise HTTPException(status_code=400, detail="Empty question")

    try:
        rag_chain = get_rag_chain(request.session_id)
        result = rag_chain.invoke({
            "question": input_text,
            "chat_history": request.chat_history
        })
        answer = result.get("answer", "").strip()

        # Clean repeated lines if necessary
        lines = answer.split("\n")
        unique_lines = []
        for line in lines:
            if line.strip() and (len(unique_lines) == 0 or line.strip() != unique_lines[-1].strip()):
                unique_lines.append(line)
        answer = "\n".join(unique_lines)

        return ChatResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
