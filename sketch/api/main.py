from fastapi import FastAPI, Cookie

from .models import SketchPad

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/api")
async def api(sketchpad: SketchPad, token: str | None = Cookie(default=None)):
    print(token)
    print(sketchpad)
