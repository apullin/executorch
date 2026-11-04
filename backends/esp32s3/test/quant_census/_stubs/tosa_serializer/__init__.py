# Copyright (c) 2026 Andrew Pullin
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A stand-in for `tosa_serializer`, so Arm's quantizers import without it.

Not the real package, and never installed: quant_census/run.py puts the
directory holding it on sys.path for the Arm and Cortex-M recipes only, and
only when the real module is missing.

What those quantizers need from tosa_serializer at import time is enumerant
names: `ts.DType.INT8` in an Enum body, `ts.DType.SHAPE` in a table, `ts.DType`
in an annotation. The stub gives every such name a distinct, hashable, inert
value. Anything that would do TOSA work -- building a serializer, a tensor or
an attribute, calling an enumerant -- raises StubUsed, so a quantizer path that
needs real TOSA functionality fails loudly (run.py then records the recipe as
skipped) instead of annotating against fakes.
"""


class StubUsed(RuntimeError):
    """Raised when code needs real tosa_serializer functionality."""


class _Enumerant:
    __slots__ = ("namespace", "name")

    def __init__(self, namespace, name):
        self.namespace = namespace
        self.name = name

    def __repr__(self):
        return f"<tosa_serializer stub {self.namespace}.{self.name}>"

    def __call__(self, *args, **kwargs):
        raise StubUsed(f"tosa_serializer.{self.namespace}.{self.name}(...) needs the "
                       f"real tosa_serializer; quant_census only stubs its names")


class _Namespace(type):
    """A class whose every public attribute is a distinct enumerant, and which
    refuses to be instantiated."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Enumerant(cls.__name__, name)
        setattr(cls, name, value)
        return value

    def __call__(cls, *args, **kwargs):
        raise StubUsed(f"tosa_serializer.{cls.__name__}(...) needs the real "
                       f"tosa_serializer; quant_census only stubs its names")


class DType(metaclass=_Namespace):
    pass


class Op(metaclass=_Namespace):
    pass


class NanPropagationMode(metaclass=_Namespace):
    pass


class RoundingMode(metaclass=_Namespace):
    pass


class ResizeMode(metaclass=_Namespace):
    pass


class TosaSerializer(metaclass=_Namespace):
    pass


class TosaSerializerTensor(metaclass=_Namespace):
    pass


class TosaSerializerAttribute(metaclass=_Namespace):
    pass


class _Names(dict):
    """DTypeNames: read only by __repr__ helpers."""

    def __missing__(self, key):
        return repr(key)


DTypeNames = _Names()
STUB = True


def __getattr__(name):
    raise AttributeError(f"tosa_serializer stub (quant_census) has no {name!r}; "
                         f"real TOSA functionality would be needed")
