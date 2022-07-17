from pydantic import BaseModel

from typing import List, Any


class SketchPadMetadata(BaseModel):
    id: str
    creation_start: str
    creation_end: str


class Sketch(BaseModel):
    name: str
    data: str


class SketchPad(BaseModel):
    version: str
    metadata: SketchPadMetadata
    sketches: List[Sketch]
    context: Any
