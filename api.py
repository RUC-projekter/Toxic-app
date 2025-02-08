from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Initialize FastAPI app
app = FastAPI()

# Load the detoxification model
MODEL_NAME = "s-nlp/bart-base-detox"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Define the request schema
class RephraseRequest(BaseModel):
    text: str

# Define the rephrase endpoint
@app.post("/rephrase")
async def rephrase_text(request: RephraseRequest):
    try:
        # Tokenize input text
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(device)

        # Generate detoxified text
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,
            num_beams=5,
            early_stopping=True,
        )

        # Decode and return the detoxified text
        rephrased_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"rephrased": rephrased_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
