from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Welcome to the Bagh Chal AI Backend"}

# Add endpoints for AI move calculations here