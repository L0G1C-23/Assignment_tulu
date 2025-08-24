from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
import joblib
import pandas as pd
from datetime import datetime
import uuid
import os
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="AI Message Triage System",
    description="Hospital message classification and ticketing system",
    version="1.0.0"
)

# Global variables for model and data storage
ml_pipeline = None
tickets_db = {}  # In-memory storage (use SQLite for production)
ticket_counter = 0

# Pydantic models
class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

class IngestRequest(BaseModel):
    from_: str = None  # Use from_ to avoid Python keyword conflict
    text: str
    
    class Config:
        # Allow field alias for 'from'
        fields = {"from_": "from"}

class TicketResponse(BaseModel):
    id: int
    from_: str = None
    text: str
    label: str
    confidence: float
    status: str
    created_at: str
    triage_required: bool
    
    class Config:
        fields = {"from_": "from"}

class TicketUpdateRequest(BaseModel):
    status: str

class TicketUpdateResponse(BaseModel):
    id: int
    status: str
    resolved_at: Optional[str] = None

class TicketListResponse(BaseModel):
    id: int
    from_: str = None
    label: str
    status: str
    
    class Config:
        fields = {"from_": "from"}

# Load ML model on startup
@app.on_event("startup")
async def load_model():
    global ml_pipeline
    try:
        ml_pipeline = joblib.load('models/model.joblib')
        # ml_pipeline = joblib.load('models/pipeline.joblib')
        print("ML model loaded successfully!")
    except Exception as e:
        print(f"Warning: Could not load ML model: {e}")
        print("Please run 'python train.py' first to train the model.")

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.post("/ml/predict", response_model=PredictionResponse)
async def predict_category(request: PredictionRequest):
    """Predict message category using ML model"""
    if ml_pipeline is None:
        raise HTTPException(status_code=500, detail="ML model not loaded. Please train the model first.")
    
    try:
        # Make prediction
        prediction = ml_pipeline.predict([request.text])[0]
        confidence_scores = ml_pipeline.predict_proba([request.text])[0]
        confidence = float(max(confidence_scores))
        
        return PredictionResponse(
            label=prediction,
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/messages/ingest", response_model=TicketResponse)
async def ingest_message(request: IngestRequest):
    """Ingest a message and create a ticket with ML classification"""
    global ticket_counter
    
    if ml_pipeline is None:
        raise HTTPException(status_code=500, detail="ML model not loaded. Please train the model first.")
    
    try:
        # Generate ticket ID
        ticket_counter += 1
        ticket_id = ticket_counter
        
        # Predict category and confidence
        prediction = ml_pipeline.predict([request.text])[0]
        confidence_scores = ml_pipeline.predict_proba([request.text])[0]
        confidence = float(max(confidence_scores))
        
        # Create ticket
        current_time = datetime.now().isoformat() + "Z"
        triage_required = confidence < 0.7
        
        ticket = {
            "id": ticket_id,
            "from": request.from_,
            "text": request.text,
            "label": prediction,
            "confidence": confidence,
            "status": "open",
            "created_at": current_time,
            "triage_required": triage_required,
            "resolved_at": None
        }
        
        # Store in database (in-memory for this implementation)
        tickets_db[ticket_id] = ticket
        
        return TicketResponse(
            id=ticket["id"],
            from_=ticket["from"],
            text=ticket["text"],
            label=ticket["label"],
            confidence=ticket["confidence"],
            status=ticket["status"],
            created_at=ticket["created_at"],
            triage_required=ticket["triage_required"]
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error ingesting message: {str(e)}")

@app.get("/tickets", response_model=List[TicketListResponse])
async def list_tickets(
    label: Optional[str] = Query(None, description="Filter by label"),
    status: Optional[str] = Query(None, description="Filter by status")
):
    """List tickets with optional filtering"""
    try:
        filtered_tickets = []
        
        for ticket in tickets_db.values():
            # Apply filters
            if label and ticket["label"] != label:
                continue
            if status and ticket["status"] != status:
                continue
                
            filtered_tickets.append(
                TicketListResponse(
                    id=ticket["id"],
                    from_=ticket["from"],
                    label=ticket["label"],
                    status=ticket["status"]
                )
            )
        
        return filtered_tickets
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error listing tickets: {str(e)}")

@app.patch("/tickets/{ticket_id}", response_model=TicketUpdateResponse)
async def resolve_ticket(ticket_id: int, request: TicketUpdateRequest):
    """Update ticket status"""
    try:
        if ticket_id not in tickets_db:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Update ticket status
        tickets_db[ticket_id]["status"] = request.status
        
        # Set resolved_at timestamp if status is resolved
        resolved_at = None
        if request.status == "resolved":
            resolved_at = datetime.now().isoformat() + "Z"
            tickets_db[ticket_id]["resolved_at"] = resolved_at
        
        return TicketUpdateResponse(
            id=ticket_id,
            status=request.status,
            resolved_at=resolved_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating ticket: {str(e)}")

# Additional endpoints for better functionality

@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: int):
    """Get a specific ticket by ID"""
    if ticket_id not in tickets_db:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    return tickets_db[ticket_id]

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not tickets_db:
        return {"total_tickets": 0, "by_status": {}, "by_label": {}}
    
    total_tickets = len(tickets_db)
    
    status_counts = {}
    label_counts = {}
    
    for ticket in tickets_db.values():
        status = ticket["status"]
        label = ticket["label"]
        
        status_counts[status] = status_counts.get(status, 0) + 1
        label_counts[label] = label_counts.get(label, 0) + 1
    
    return {
        "total_tickets": total_tickets,
        "by_status": status_counts,
        "by_label": label_counts,
        "model_loaded": ml_pipeline is not None
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"detail": "Endpoint not found"}

@app.exception_handler(500)
async def internal_server_error_handler(request, exc):
    return {"detail": "Internal server error"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)