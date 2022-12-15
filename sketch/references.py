import hashlib
import json
import os

try:
    from functools import cache
except ImportError:
    from functools import lru_cache

    cache = lru_cache(maxsize=None)
from typing import Dict


def get_id_for_object(obj):
    serialized = json.dumps(obj, sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class Reference:
    def __init__(self, **data):
        self.data = data
        self.id = get_id_for_object(self.data)
        self.type = self.__class__.__name__

    def to_pyscript(self):
        raise NotImplementedError(f"{self.__class__}.to_usable_script")

    def to_searchable_string(self):
        raise NotImplementedError(f"{self.__class__}.to_searchable_string")

    def to_dict(self):
        return {
            "id": self.id,
            "type": self.type,
            "data": self.data,
        }

    def to_json(self):
        return json.dumps(self.to_dict())

    @classmethod
    @property
    @cache
    def subclass_lookup(cls) -> Dict[str, "Reference"]:
        subclasses = {}
        for subclass in cls.__subclasses__():
            subclasses[subclass.__name__] = subclass
            subclasses.update(subclass.subclass_lookup)
        return subclasses

    @classmethod
    def from_dict(cls, data):
        subclass = cls.subclass_lookup[data["type"]]
        new_obj = subclass(**data["data"])
        assert new_obj.id == data["id"]
        return new_obj

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls.from_dict(data)

    @property
    def short_id(self):
        return int.from_bytes(bytes.fromhex(self.id[:16]), "big", signed=True)


# TODO: make the subclasses of Reference have smarter args
# possibly make them a dataclass

# TODO: eventually consider a Sqlite Query reference (full tuple)
# might replace this entire single column concept
class SqliteColumn(Reference):
    def __init__(self, path, query, column, friendly_name=None):
        data = {
            "path": path,
            "column": column,
            "query": query,
            "friendly_name": friendly_name,
        }
        super().__init__(**data)

    def to_searchable_string(self):
        base = f"{self.data['query']} {self.data['column']}"
        base += f" {self.data['friendly_name']}" if self.data["friendly_name"] else ""
        base += f" {self.data['path']}"
        return base

    def to_pyscript(self):
        commands = ["import os", "import pandas as pd", "import sqlite3"]
        if self.data["path"].startswith("http"):
            # assuming this is a downloadable path
            commands.append(
                f"""os.system("wget -nc '{self.data['path']}' -P ~/.cache/sketch/")"""  # noqa
            )
            base = os.path.split(self.data["path"])[1]
            localpath = f"~/.cache/sketch/{base}"
        else:
            localpath = self.data["path"]
        commands.append(f"conn = sqlite3.connect('{localpath}')")
        commands.append(f"df = pd.read_sql_query('{self.data['query']}', conn)")
        commands.append(f"df = df['{self.data['column']}']")
        return "\n".join(commands)


class PandasDataframeColumn(Reference):
    def __init__(self, column, dfname, **dfextra):
        super().__init__(dfname=dfname, column=column, **dfextra)

    def to_searchable_string(self):
        base = " ".join([self.data["dfname"], self.data["column"]])
        base += " ".join([f"{k}={v}" for k, v in self.data.get("extra", {}).items()])
        return base

    def to_pyscript(self):
        commands = []
        commands.append(f'df = {self.data["dfname"]}')
        commands.append(f'df = df[["{self.data["column"]}"]]')
        return "\n".join(commands)


class WikipediaTableColumn(Reference):
    def __init__(self, page, id, headers, column):
        super().__init__(page=page, id=id, headers=headers, column=column)

    def to_searchable_string(self):
        base = " ".join(
            [
                self.data["page"],
                str(self.data["id"]),
                self.data["headers"],
                str(self.data["column"]),
            ]
        )
        return base

    @property
    def url(self):
        return f"https://en.wikipedia.org/wiki/{self.data['page'].replace(' ', '_')}"

    def to_pyscript(self):
        commands = []
        commands.append(f"import pandas as pd")
        commands.append(f'df = pd.read_html({self.data["page"]})[{self.data["id"]}]')
        commands.append(f'df = df[["{self.data["column"]}"]]')
        return "\n".join(commands)
