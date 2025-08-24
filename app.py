from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional, List
import joblib
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Global variables for model and data storage
vectorizer = None
model = None
tickets_db = {}  # In-memory dictionary (SQLite preferred but this works)
ticket_counter = 0

# Pydantic models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class IngestRequest(BaseModel):
    from_: str = Field(alias="from")
    text: str

class TicketResponse(BaseModel):
    id: int
    from_: str = Field(alias="from")
    text: str
    label: str
    confidence: float
    status: str
    created_at: str
    triage_required: bool

class TicketUpdateRequest(BaseModel):
    status: str

class TicketUpdateResponse(BaseModel):
    id: int
    status: str
    resolved_at: Optional[str] = None

class TicketListItem(BaseModel):
    id: int
    from_: str = Field(alias="from")
    label: str
    status: str

# Load ML model on startup
@app.on_event("startup")
async def load_model():
    global vectorizer, model
    try:
        # Load the trained components from train.py
        vectorizer = joblib.load('models/vectorizer.joblib')
        model = joblib.load('models/model.joblib')
        print("ML model loaded successfully!")
    except Exception as e:
        print(f"Could not load model: {e}")
        print("Please run 'python train.py' first to train the model.")

# 1) Health Check
@app.get("/health")
async def health_check():
    return {"status": "ok"}

# 2) Predict Category (ML model)
@app.post("/ml/predict", response_model=PredictionResponse)
async def predict_category(request: PredictionRequest):
    if vectorizer is None or model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded. Please train the model first.")
    
    try:
        # Transform text using the trained vectorizer
        text_tfidf = vectorizer.transform([request.text])
        
        # Make prediction using the trained classifier
        prediction = model.predict(text_tfidf)[0]
        confidence_scores = model.predict_proba(text_tfidf)[0]
        confidence = float(max(confidence_scores))
        
        return PredictionResponse(label=prediction, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# 3) Ingest Message (create ticket)
@app.post("/messages/ingest", response_model=TicketResponse)
async def ingest_message(request: IngestRequest):
    global ticket_counter
    
    if vectorizer is None or model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded. Please train the model first.")
    
    try:
        # Generate ticket ID (auto increment)
        ticket_counter += 1
        ticket_id = ticket_counter
        
        # Use ML model to predict label + confidence
        text_tfidf = vectorizer.transform([request.text])
        prediction = model.predict(text_tfidf)[0]
        confidence_scores = model.predict_proba(text_tfidf)[0]
        confidence = float(max(confidence_scores))
        
        # Create ticket with all required fields
        current_time = datetime.now().isoformat() + "Z"
        
        # If confidence < 0.7 → return "triage_required": true
        triage_required = confidence < 0.7
        
        ticket = {
            "id": ticket_id,
            "from": request.from_,
            "text": request.text,
            "label": prediction,
            "confidence": confidence,
            "status": "open",
            "created_at": current_time,
            "triage_required": triage_required
        }
        
        # Store ticket in in-memory dictionary
        tickets_db[ticket_id] = ticket
        
        return TicketResponse(**ticket)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting message: {str(e)}")

# 4) List Tickets
@app.get("/tickets", response_model=List[TicketListItem])
async def list_tickets(
    label: Optional[str] = Query(None, description="Filter by label"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    try:
        filtered_tickets = []
        
        for ticket in tickets_db.values():
            # Apply filters
            if label and ticket["label"] != label:
                continue
            if status and ticket["status"] != status:
                continue
                
            filtered_tickets.append(TicketListItem(
                id=ticket["id"],
                **{"from": ticket["from"]},
                label=ticket["label"],
                status=ticket["status"]
            ))
        
        return filtered_tickets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tickets: {str(e)}")

# 5) Resolve Ticket
@app.patch("/tickets/{id}", response_model=TicketUpdateResponse)
async def resolve_ticket(id: int, request: TicketUpdateRequest):
    try:
        if id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Update ticket status
        tickets_db[id]["status"] = request.status
        
        # Set resolved_at timestamp if status is resolved
        resolved_at = None
        if request.status == "resolved":
            resolved_at = datetime.now().isoformat() + "Z"
            tickets_db[id]["resolved_at"] = resolved_at
        
        return TicketUpdateResponse(
            id=id,
            status=request.status,
            resolved_at=resolved_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating ticket: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)