import base64

import datasketch

# import datasketches


def active(func):
    def wrapper(self, *args, **kwargs):
        assert self.active, "Sketchpad is not active, cannot add a row"
        return func(self, *args, **kwargs)

    return wrapper


class SketchBase:
    def __init__(self, data, active=False):
        self.name = self.__class__.__name__
        self.data = data
        self.active = active

    @active
    def add_row(self, row):
        raise NotImplementedError(f"{self.__class__.__name__}.add_row")

    @classmethod
    def from_series(cls, series):
        raise NotImplementedError(f"{cls.__name__}.from_series")

    def pack(self):
        return self.data

    @staticmethod
    def unpack(data):
        return data

    def to_dict(self):
        return {"name": self.__class__.__name__, "data": self.pack()}

    @classmethod
    def empty_data(cls):
        raise NotImplementedError(f"{cls.__name__}.empty_data")

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

    @classmethod
    def empty(cls):
        return cls(data=cls.empty_data(), active=True)

    def freeze(self):
        self.active = False

    def merge(self, sketch):
        raise NotImplementedError(f"{self.__class__.__name__}.merge")


class Rows(SketchBase):
    @active
    def add_row(self, row):
        self.data += 1

    @classmethod
    def from_series(cls, series):
        return cls(data=int(series.size))

    @classmethod
    def empty_data(cls):
        return 0


class Count(SketchBase):
    @active
    def add_row(self, row):
        self.data += 1 if row is not None else 0

    @classmethod
    def from_series(cls, series):
        return cls(data=int(series.count()))

    @classmethod
    def empty_data(cls):
        return 0


class MinHash(SketchBase):
    @active
    def add_row(self, row):
        # TODO: ensure row is 'bytes'
        self.data.update(str(row).encode("utf-8"))

    @classmethod
    def from_series(cls, series):
        minhash = datasketch.MinHash()
        minhash.update_batch([str(x).encode("utf-8") for x in series])
        lmh = datasketch.LeanMinHash(minhash)
        return cls(data=lmh)

    def pack(self):
        if self.active:
            raise RuntimeError("Cannot pack an active MinHash")
        buf = bytearray(self.data.bytesize())
        self.data.serialize(buf)
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def unpack(data):
        return datasketch.LeanMinHash.deserialize(base64.b64decode(data))

    @classmethod
    def empty_data(cls):
        return datasketch.MinHash()

    def freeze(self):
        self.data = datasketch.LeanMinHash(self.data)
        super().freeze()


class HyperLogLog(SketchBase):
    @active
    def add_row(self, row):
        # TODO: ensure row is 'bytes'
        self.data.update(str(row).encode("utf-8"))

    @classmethod
    def from_series(cls, series):
        hllpp = datasketch.HyperLogLogPlusPlus()
        for d in series:
            hllpp.update(str(d).encode("utf-8"))
        return cls(data=hllpp)

    def pack(self):
        buf = bytearray(self.data.bytesize())
        self.data.serialize(buf)
        return base64.b64encode(buf).decode("utf-8")

    @staticmethod
    def unpack(data):
        return datasketch.HyperLogLogPlusPlus.deserialize(base64.b64decode(data))

    @classmethod
    def empty_data(cls):
        return datasketch.HyperLogLogPlusPlus()
