from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# Load the detoxification model
MODEL_NAME = "s-nlp/bart-base-detox"
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class RephraseRequest(BaseModel):
    text: str

@app.post("/rephrase")
async def rephrase_text(request: RephraseRequest):
    try:
        # Use the model to detoxify the text
        inputs = tokenizer(
            request.text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
        ).to(model.device)

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
        raise HTTPException(status_code=500, detail=str(e))
