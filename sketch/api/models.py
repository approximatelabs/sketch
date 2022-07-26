from typing import Any, Dict, List

from pydantic import BaseModel

# These API models are specifically a layer for:
# 1. Defining the API for the models in Documentation
# 2. Validating the data that comes in from the client
# 3. Converting the data to the correct type in python (json -> python)

# In order to actually use a SketchPad object, convert to a SketchPad object
# from ..core import SketchPad


# TODO: Consider validation on hash(data) == id
class SourceMetadata(BaseModel):
    id: str
    data: Dict[str, Any]


# TODO: Consider validation on hash(data) == id
class RelationMetadata(BaseModel):
    id: str
    data: Dict[str, Any]


# TODO: Consider validation on hash(data) == id
class ReferenceMetadata(BaseModel):
    id: str
    data: Dict[str, Any]


class SketchPadMetadata(BaseModel):
    id: str
    creation_start: str
    creation_end: str
    stream_id: str | None = None
    source: SourceMetadata | None = None
    relation: RelationMetadata | None = None
    reference: ReferenceMetadata | None = None


class Sketch(BaseModel):
    name: str
    data: str


class SketchPad(BaseModel):
    version: str
    metadata: SketchPadMetadata
    sketches: List[Sketch]
    context: Dict[str, Any]
