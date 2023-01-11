import base64
import json

import datasketch
import datasketches


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
        result = cls(data=cls.empty_data(), active=True)
        for d in series:
            result.add_row(d)
        return result

    def pack(self):
        return self.data

    @classmethod
    def unpack(cls, data):
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
            for subclass in cls.all_sketches():
                if subclass.__name__ == data["name"]:
                    tcls = subclass
        return tcls(data=tcls.unpack(data["data"]))

    @classmethod
    def all_sketches(cls):
        subclasses = cls.__subclasses__()
        for subclass in list(subclasses):
            subclasses.extend(subclass.all_sketches())
        # filter
        subclasses = [s for s in subclasses if s.__name__ != "DataSketchesSketchBase"]
        return subclasses

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

    @classmethod
    def unpack(cls, data):
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

    @classmethod
    def unpack(cls, data):
        return datasketch.HyperLogLogPlusPlus.deserialize(base64.b64decode(data))

    @classmethod
    def empty_data(cls):
        return datasketch.HyperLogLogPlusPlus()


class DataSketchesSketchBase(SketchBase):
    sketch_class = None
    init_args = ()

    @active
    def add_row(self, row):
        self.data.update(str(row).encode("utf-8"))

    def pack(self):
        return base64.b64encode(self.data.serialize()).decode("utf-8")

    @classmethod
    def unpack(cls, data):
        return cls.sketch_class.deserialize(base64.b64decode(data))

    @classmethod
    def empty_data(cls):
        return cls.sketch_class(*cls.init_args)


class DS_HLL(DataSketchesSketchBase):
    sketch_class = datasketches.hll_sketch
    init_args = (12, datasketches.tgt_hll_type.HLL_8)

    def pack(self):
        return base64.b64encode(self.data.serialize_compact()).decode("utf-8")


class DS_CPC(DataSketchesSketchBase):
    sketch_class = datasketches.cpc_sketch
    init_args = (12,)


class DS_FI(DataSketchesSketchBase):
    sketch_class = datasketches.frequent_strings_sketch
    init_args = (10,)


class DS_KLL(DataSketchesSketchBase):
    sketch_class = datasketches.kll_floats_sketch
    init_args = (160,)

    @active
    def add_row(self, row):
        if isinstance(row, (int, float)):
            self.data.update(row)


class DS_Quantiles(DataSketchesSketchBase):
    sketch_class = datasketches.quantiles_floats_sketch
    init_args = (128,)

    @active
    def add_row(self, row):
        if isinstance(row, (int, float)):
            self.data.update(row)


class DS_REQ(DataSketchesSketchBase):
    sketch_class = datasketches.req_floats_sketch
    init_args = (12,)

    @active
    def add_row(self, row):
        if isinstance(row, (int, float)):
            self.data.update(row)


class DS_THETA(DataSketchesSketchBase):
    sketch_class = datasketches.update_theta_sketch
    init_args = (12,)

    def pack(self):
        try:
            return base64.b64encode(self.data.compact().serialize()).decode("utf-8")
        except AttributeError:
            return base64.b64encode(self.data.serialize()).decode("utf-8")

    @classmethod
    def unpack(cls, data):
        return datasketches.compact_theta_sketch.deserialize(base64.b64decode(data))


class PyUnicodeStringsSerDe(datasketches.PyObjectSerDe):
    def get_size(self, item):
        return int(4 + len(item.encode("utf-8")))

    def to_bytes(self, item: str):
        b = bytearray()
        b.extend(len(item.encode("utf-8")).to_bytes(4, "little"))
        b.extend(item.encode("utf-8"))
        return bytes(b)

    def from_bytes(self, data: bytes, offset: int):
        num_chars = int.from_bytes(data[offset : offset + 3], "little")
        if num_chars < 0 or num_chars > offset + len(data):
            raise IndexError(
                f"num_chars read must be non-negative and not larger than the buffer. Found {num_chars}"
            )
        str = data[offset + 4 : offset + 4 + num_chars].decode("utf-8")
        return (str, 4 + num_chars)


class DS_VO(DataSketchesSketchBase):
    sketch_class = datasketches.var_opt_sketch
    init_args = (50,)

    @active
    def add_row(self, row):
        self.data.update(str(row))

    def pack(self):
        return base64.b64encode(self.data.serialize(PyUnicodeStringsSerDe())).decode(
            "utf-8"
        )

    @classmethod
    def unpack(cls, data):
        return cls.sketch_class.deserialize(
            base64.b64decode(data), PyUnicodeStringsSerDe()
        )


class UnicodeMatches(SketchBase):
    unicode_ranges = {
        "emoticon": (0x1F600, 0x1F64F),
        "control": (0x00, 0x1F),
        "digits": (0x30, 0x39),
        "latin-lower": (0x41, 0x5A),
        "latin-upper": (0x61, 0x7A),
        "basic-latin": (0x00, 0x7F),
        "extended-latin": (0x0080, 0x02AF),
        "UNKNOWN": (0x00, 0x00),
    }

    @active
    def add_row(self, row):
        if isinstance(row, str):
            for c in row:
                found = False
                for name, (start, end) in self.unicode_ranges.items():
                    if start <= ord(c) <= end:
                        self.data[name] += 1
                        found = True
                if not found:
                    self.data["UNKNOWN"] += 1

    def pack(self):
        return base64.b64encode(json.dumps(self.data).encode("utf-8")).decode("utf-8")

    @classmethod
    def unpack(cls, data):
        return json.loads(base64.b64decode(data))

    @classmethod
    def empty_data(cls):
        return {name: 0 for name in cls.unicode_ranges}
