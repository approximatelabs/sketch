import base64
import datetime
import heapq
import logging
import sqlite3
import uuid

import pandas as pd

import sketch

from .sketches import SketchBase


class SketchPad:
    version = "0.0.1"
    sketches = SketchBase.all_sketches()

    def __init__(self, context=None):
        self.version = "0.0.1"
        self.id = str(uuid.uuid4())
        self.metadata = {
            "id": self.id,
            "creation_start": datetime.datetime.utcnow().isoformat(),
        }
        self.context = context or {}
        # TODO: consider alternate naming convention
        # so can do dictionary lookups
        self.sketches = []

    @classmethod
    def from_series(cls, series, context=None):
        sp = cls(context=context)
        for skcls in cls.sketches:
            sp.sketches.append(skcls.from_series(series))
        sp.metadata["creation_end"] = datetime.datetime.utcnow().isoformat()
        sp.context["column_name"] = series.name
        return sp

    def get_sketch_by_name(self, name):
        sketches = [sk for sk in self.sketches if sk.name == name]
        if len(sketches) == 1:
            return sketches[0]
        return None

    def get_sketchdata_by_name(self, name):
        sketch = self.get_sketch_by_name(name)
        return sketch.data if sketch else None

    def minhash_jaccard(self, other):
        self_minhash = self.get_sketchdata_by_name("MinHash")
        other_minhash = other.get_sketchdata_by_name("MinHash")
        if self_minhash is None or other_minhash is None:
            return None
        return self_minhash.jaccard(other_minhash)

    def to_dict(self):
        return {
            "version": self.version,
            "metadata": self.metadata,
            "sketches": [s.to_dict() for s in self.sketches],
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data):
        assert data["version"] == cls.version
        sp = cls()
        sp.id = data["metadata"]["id"]
        sp.metadata = data["metadata"]
        sp.context = data["context"]
        sp.sketches = [SketchBase.from_dict(s) for s in data["sketches"]]
        return sp


class Portfolio:
    def __init__(self, sketchpads=None):
        self.sketchpads = {sp.id: sp for sp in (sketchpads or [])}

    @classmethod
    def from_dataframe(cls, df):
        return cls().add_dataframe(df)

    def add_dataframe(self, df):
        for col in df.columns:
            sp = SketchPad.from_series(df[col], context=df.attrs)
            self.add_sketchpad(sp)
        return self

    @classmethod
    def from_dataframes(cls, dfs):
        return cls().add_dataframes(dfs)

    def add_dataframes(self, dfs):
        for df in dfs:
            self.add_dataframe(df)
        return self

    @classmethod
    def from_sketchpad(cls, sketchpad):
        return cls().add_sketchpad(sketchpad)

    def add_sketchpad(self, sketchpad):
        self.sketchpads[sketchpad.id] = sketchpad
        return self

    @classmethod
    def from_sqlite(cls, sqlite_db_path):
        return cls().add_sqlite(sqlite_db_path)

    def add_sqlite(self, sqlite_db_path):
        conn = sqlite3.connect(sqlite_db_path)
        tables = pd.read_sql(
            "SELECT name FROM sqlite_schema WHERE type='table' ORDER BY name;", conn
        )
        logging.info(f"Found {len(tables)} tables in file {sqlite_db_path}")
        all_tables = {}
        for i, table in enumerate(tables.name):
            df = pd.read_sql(f"SELECT * from '{table}'", conn)
            df.attrs |= {"table_name": table, "source": sqlite_db_path}
            self.add_dataframe(df)
        return self

    def closest_overlap(self, sketchpad, n=5):
        scores = []
        for sp in self.sketchpads.values():
            score = sketchpad.minhash_jaccard(sp)
            heapq.heappush(scores, (score, sp.id))
        top_n = heapq.nlargest(n, scores, key=lambda x: x[0])
        return [(s, self.sketchpads[i]) for s, i in top_n]
