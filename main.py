from fastapi import FastAPI

app = FastAPI(title="CloudPulse API")

@app.get("/")
def read_root():
    return {"message": "CloudPulse API running"}

@app.get("/docs")
def docs():
    return {"docs": "API documentation at /docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
