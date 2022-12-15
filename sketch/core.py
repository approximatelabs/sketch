import datetime
import heapq
import logging
import os
import sqlite3
import uuid

import pandas as pd
from packaging import version

from .metrics import binary_metrics, strings_from_sketchpad_sketches, unary_metrics
from .references import (
    PandasDataframeColumn,
    Reference,
    SqliteColumn,
    WikipediaTableColumn,
)
from .sketches import SketchBase

SKETCHCACHE = "~/.cache/sketch/"

# TODO: These object models are possibly different than the ones in api models
# and those are different than the ones in data models... need to rectify.
# either use a single source of truth, or have good robust tests.
# -- These feel more useful for the client, utility methods


# TODO: consider sketchpad having the same "interface" as a sketch..
# maybe that's the "abstraction" here...
class SketchPad:
    version = "0.0.1"
    sketch_classes = SketchBase.all_sketches()

    def __init__(self, reference, context=None, initialize_sketches=True):
        self.version = "0.0.1"
        self.id = str(uuid.uuid4())
        self.metadata = {
            "id": self.id,
            "creation_start": datetime.datetime.utcnow().isoformat(),
        }
        self.reference = reference
        self.context = context or {}
        if initialize_sketches:
            self.sketches = [skcls.empty() for skcls in self.sketch_classes]
        else:
            self.sketches = []

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

    def compute_sketches(self, data):
        # data is assumed to be an iterable
        for row in data:
            for sk in self.sketches:
                sk.add_row(row)
        # freeze sketches
        for sk in self.sketches:
            sk.freeze()

    def to_dict(self):
        return {
            "version": self.version,
            "metadata": self.metadata,
            "reference": self.reference.to_dict(),
            "sketches": [s.to_dict() for s in self.sketches],
            "context": self.context,
        }

    @classmethod
    def from_series(cls, series: pd.Series, reference: Reference = None) -> "SketchPad":
        if reference is None:
            reference = PandasDataframeColumn("df", series.name)
        sp = cls(reference, initialize_sketches=False)
        for skcls in cls.sketch_classes:
            sp.sketches.append(skcls.from_series(series))
        sp.metadata["creation_end"] = datetime.datetime.utcnow().isoformat()
        return sp

    @classmethod
    def from_dict(cls, data):
        assert data["version"] == cls.version
        sp = cls(Reference.from_dict(data["reference"]))
        sp.id = data["metadata"]["id"]
        sp.metadata = data["metadata"]
        sp.context = data["context"]
        sp.sketches = [SketchBase.from_dict(s) for s in data["sketches"]]
        return sp

    def get_metrics(self):
        return unary_metrics(self)

    def get_cross_metrics(self, other):
        return binary_metrics(self, other)

    def string_value_representation(self):
        return strings_from_sketchpad_sketches(self)


class Portfolio:
    def __init__(self, sketchpads=None):
        self.sketchpads = {sp.id: sp for sp in (sketchpads or [])}

    @classmethod
    def from_dataframe(cls, df, dfname="df"):
        return cls().add_dataframe(df, dfname=dfname)

    def add_dataframe(self, df, dfname="df"):
        for col in df.columns:
            reference = PandasDataframeColumn(dfname, col)
            sp = SketchPad.from_series(df[col], reference)
            self.add_sketchpad(sp)
        return self

    @classmethod
    def from_dataframes(cls, dfs):
        return cls().add_dataframes(dfs)

    def add_dataframes(self, dfs):
        # in general, this method is poor because of name tracking
        for df in dfs:
            self.add_dataframe(df)
        return self

    @classmethod
    def from_sqlite(cls, sqlite_db_path):
        return cls().add_sqlite(sqlite_db_path)

    def add_wikitable(self, page, id, headers, pandas_df):
        for col in pandas_df.columns:
            reference = WikipediaTableColumn(page, id, headers, col)
            sp = SketchPad.from_series(pandas_df[col], reference)
            self.add_sketchpad(sp)

    def get_sketchpad_by_reference_id(self, reference_id):
        for sketchpad in self.sketchpads.values():
            if sketchpad.reference.id == reference_id:
                return sketchpad
        return None

    def add_sqlite(self, sqlite_db_path):
        if sqlite_db_path.startswith("http"):
            os.system(f"wget -nc {sqlite_db_path} --directory-prefix={SKETCHCACHE} -q")
            path = os.path.join(SKETCHCACHE, os.path.split(sqlite_db_path)[1])
        else:
            path = sqlite_db_path
        conn = sqlite3.connect(path)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        # TODO: Consider using a cursor to avoid the need for this
        meta_name = (
            "sqlite_master"
            if version.parse(sqlite3.sqlite_version) < version.Version("3.33.0")
            else "sqlite_schema"
        )
        tables = pd.read_sql(
            f"SELECT name FROM {meta_name} WHERE type='table' ORDER BY name;", conn
        )
        logging.info(f"Found {len(tables)} tables in file {sqlite_db_path}")
        for i, table in enumerate(tables.name):
            for column in pd.read_sql(f'PRAGMA table_info("{table}")', conn).name:
                query = f'SELECT "{column}" FROM "{table}"'
                reference = SqliteColumn(sqlite_db_path, query, column)
                # consider iterator here
                sp = SketchPad.from_series(
                    pd.read_sql(query, conn)[f"{column}"],
                    reference,
                )
                self.add_sketchpad(sp)
        return self

    @classmethod
    def from_sketchpad(cls, sketchpad):
        return cls().add_sketchpad(sketchpad)

    def add_sketchpad(self, sketchpad):
        self.sketchpads[sketchpad.id] = sketchpad
        return self

    def get_approx_pk_sketchpads(self):
        # is an estimated unique_key if unique count estimate
        # is > 97% the number of rows
        pf = Portfolio()
        for sketchpad in self.sketchpads.values():
            uq = sketchpad.get_sketchdata_by_name("HyperLogLog").count()
            rows = int(sketchpad.get_sketchdata_by_name("Rows"))
            if uq > 0.97 * rows:
                pf.add_sketchpad(sketchpad)
        return pf

    def closest_overlap(self, sketchpad, n=5):
        scores = []
        for sp in self.sketchpads.values():
            score = sketchpad.minhash_jaccard(sp)
            heapq.heappush(scores, (score, sp.id))
        top_n = heapq.nlargest(n, scores, key=lambda x: x[0])
        return [(s, self.sketchpads[i]) for s, i in top_n]
