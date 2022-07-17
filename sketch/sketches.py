import base64
import datasketch

# import datasketches


class SketchBase:
    def __init__(self, data):
        self.name = self.__class__.__name__
        self.data = data

    @classmethod
    def from_series(cls, series):
        raise NotImplementedError(f"Need from_series method for {cls.__name__}")

    def pack(self):
        return self.data

    @staticmethod
    def unpack(data):
        return data

    def to_dict(self):
        return {"name": self.__class__.__name__, "data": self.pack()}

    @classmethod
    def from_dict(cls, data):
        tcls = cls
        if data["name"] != cls.__name__:
            for subclass in cls.__subclasses__():
                if subclass.__name__ == data["name"]:
                    tcls = subclass
        return tcls(data=tcls.unpack(data["data"]))

    @classmethod
    def all_sketches(cls):
        return list(cls.__subclasses__())


class Rows(SketchBase):
    @classmethod
    def from_series(cls, series):
        return cls(data=int(series.size))


class Count(SketchBase):
    @classmethod
    def from_series(cls, series):
        return cls(data=int(series.count()))


class MinHash(SketchBase):
    @classmethod
    def from_series(cls, series):
        minhash = datasketch.MinHash()
        minhash.update_batch([str(x).encode("utf-8") for x in series])
        lmh = datasketch.LeanMinHash(minhash)
        return cls(data=lmh)

    def pack(self):
        buf = bytearray(self.data.bytesize())
        self.data.serialize(buf)
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def unpack(data):
        return datasketch.LeanMinHash.deserialize(base64.b64decode(data))
