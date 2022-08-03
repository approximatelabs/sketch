import hashlib
import json
from functools import cache
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


# TODO: make the subclasses of Reference have smarter args
# possibly make them a dataclass


class SqliteColumn(Reference):
    def __init__(self, path, column, table=None, query=None, friendly_name=None):
        if table is None:
            assert query is not None
        if query is None:
            query = f"SELECT {column} FROM {table}"
        data = {
            "path": path,
            "column": column,
            "query": query,
            "table": table,
            "friendly_name": friendly_name,
        }
        super().__init__(**data)

    def to_searchable_string(self):
        base = f"{self.data['query']} {self.data['column']}"
        base += f" {self.data['friendly_name']}" if self.data["friendly_name"] else ""
        base += f" {self.data['path']}"
        return base

    def to_pyscript(self):
        commands = []
        if self.data["path"].startswith("http"):
            # assuming this is a downloadable path
            commands.append(
                f"""import os; os.system("wget {self.data['path']} -O ~/.cache/sketch/{self.id}")"""
            )  # noqa
            localpath = f"~/.cache/sketch/{self.id}"
        else:
            localpath = self.data["path"]
        commands.append(
            f"""
            conn = sqlite3.connect({localpath})
        """
        )
        commands.append(
            f"""
            df = pd.read_sql_query({self.data['query']}, conn)
        """
        )
        commands.append(
            f"""
            series = df[{self.data['column']}]
        """
        )
        return "\n".join(commands)


class PandasDataframeColumn(Reference):
    def __init__(self, dfname, column, **dfextra):
        super().__init__(dfname=dfname, column=column, extra=dfextra)

    def to_searchable_string(self):
        base = " ".join([self.data["dfname"], self.data["column"]])
        base += " ".join([f"{k}={v}" for k, v in self.data["extra"].items()])
        return base

    def to_pyscript(self):
        commands = []
        commands.append(
            f"""
            df = {self.data["dfname"]}
        """
        )
        commands.append(
            f"""
            series = df[{self.data["column"]}]
        """
        )
        return "\n".join(commands)
