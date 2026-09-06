"""Qt-free lossless Parquet codec for frozen numerical figure resources."""
from __future__ import annotations

import io
import json

import numpy as np
import pandas as pd

_TABLE_CODEC_KEY = b"nanoscope.figure_table_codec"


def _json_bytes(value):
    return json.dumps(value, allow_nan=False, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _object_scalar(value) -> list:
    """Restricted, reversible scalar tags; never pickle or execute stored objects."""
    if value is None:
        return ["none"]
    if value is pd.NA:
        return ["pd.NA"]
    if value is pd.NaT:
        return ["pd.NaT"]
    if isinstance(value, np.generic):
        if value.dtype.kind not in "biufUS":
            raise TypeError(f"Cannot losslessly encode object scalar {value.dtype}.")
        scalar = value.item()
        if isinstance(scalar, np.generic):
            raise TypeError(f"Cannot losslessly encode object scalar {value.dtype}.")
        return ["numpy", value.dtype.str, _object_scalar(scalar)]
    if type(value) is bool:
        return ["bool", value]
    if type(value) is int:
        return ["int", str(value)]
    if type(value) is float:
        return ["float", value.hex()]
    if type(value) is str:
        return ["str", value]
    if type(value) is bytes:
        return ["bytes", value.hex()]
    raise TypeError(f"Cannot losslessly encode object scalar {type(value).__name__}.")


def _restore_object_scalar(value):
    if not isinstance(value, list) or not value or not isinstance(value[0], str):
        raise ValueError("Invalid frozen table scalar encoding.")
    tag = value[0]
    if len(value) == 1 and tag in {"none", "pd.NA", "pd.NaT"}:
        return {"none": None, "pd.NA": pd.NA, "pd.NaT": pd.NaT}[tag]
    if tag == "numpy" and len(value) == 3 and isinstance(value[1], str):
        dtype = np.dtype(value[1])
        if dtype.kind not in "biufUS" or dtype.itemsize > 1_000_000:
            raise ValueError("Invalid frozen NumPy scalar type.")
        scalar = _restore_object_scalar(value[2])
        expected = {"b": bool, "i": int, "u": int, "f": float, "U": str, "S": bytes}[dtype.kind]
        if type(scalar) is not expected:
            raise ValueError("Frozen NumPy scalar tag does not match its dtype.")
        if dtype.kind in "iu" and not np.iinfo(dtype).min <= scalar <= np.iinfo(dtype).max:
            raise ValueError("Frozen NumPy scalar is outside its dtype range.")
        restored = dtype.type(scalar)
        if _object_scalar(restored) != value:
            raise ValueError("Frozen NumPy scalar cannot be reconstructed exactly.")
        return restored
    if len(value) == 2:
        if tag == "bool" and type(value[1]) is bool:
            return value[1]
        if isinstance(value[1], str):
            if tag == "int":
                return int(value[1])
            if tag == "float":
                return float.fromhex(value[1])
            if tag == "str":
                return value[1]
            if tag == "bytes":
                return bytes.fromhex(value[1])
    raise ValueError("Invalid frozen table scalar encoding.")


def _needs_object_codec(values) -> bool:
    # Arrow already retains plain string/None object columns faithfully. Mixed
    # scalars, object-typed numbers and sentinel distinctions need explicit tags.
    return any(value is not None and type(value) is not str for value in values)


def _table_parquet_bytes(data: pd.DataFrame) -> bytes:
    import pyarrow as pa
    import pyarrow.parquet as pq

    encoded = data.copy(deep=True)
    columns = []
    for position in range(len(data.columns)):
        series = data.iloc[:, position]
        if series.dtype == object and _needs_object_codec(series):
            values = [_json_bytes(_object_scalar(value)).decode("utf-8") for value in series]
            encoded.isetitem(position, pd.Series(values, index=encoded.index, dtype=object))
            columns.append(position)
    index_levels = []
    indexes = []
    for level in range(data.index.nlevels):
        values = data.index.get_level_values(level)
        if values.dtype == object and _needs_object_codec(values):
            indexes.append(pd.Index([_json_bytes(_object_scalar(value)).decode("utf-8")
                                     for value in values], dtype=object, name=values.name))
            index_levels.append(level)
        else:
            indexes.append(values)
    if index_levels:
        encoded.index = (indexes[0] if data.index.nlevels == 1 else
                         pd.MultiIndex.from_arrays(indexes, names=data.index.names))
    table = pa.Table.from_pandas(encoded, preserve_index=True)
    metadata = dict(table.schema.metadata or {})
    metadata[_TABLE_CODEC_KEY] = _json_bytes({"version": 1, "object_columns": columns,
                                             "object_index_levels": index_levels,
                                             "codec": "tagged-json-scalars"})
    buffer = io.BytesIO()
    pq.write_table(table.replace_schema_metadata(metadata), buffer)
    return buffer.getvalue()


def _table_from_parquet(payload: bytes) -> pd.DataFrame:
    import pyarrow.parquet as pq

    table = pq.read_table(io.BytesIO(payload))
    data = table.to_pandas()
    codec = (table.schema.metadata or {}).get(_TABLE_CODEC_KEY)
    if codec is None:
        return data
    metadata = json.loads(codec)
    if (not isinstance(metadata, dict) or type(metadata.get("version")) is not int
            or metadata["version"] != 1 or metadata.get("codec") != "tagged-json-scalars"):
        raise ValueError("Unsupported frozen table object encoding.")
    columns, index_levels = metadata.get("object_columns"), metadata.get("object_index_levels")
    for values, limit in ((columns, len(data.columns)), (index_levels, data.index.nlevels)):
        if (not isinstance(values, list) or any(type(i) is not int or not 0 <= i < limit for i in values)
                or len(values) != len(set(values))):
            raise ValueError("Invalid frozen table object-column metadata.")
    for position in columns:
        values = [_restore_object_scalar(json.loads(value)) for value in data.iloc[:, position]]
        data.isetitem(position, pd.Series(values, index=data.index, dtype=object))
    if index_levels:
        indexes = []
        for level in range(data.index.nlevels):
            values = data.index.get_level_values(level)
            if level in index_levels:
                values = pd.Index([_restore_object_scalar(json.loads(value)) for value in values],
                                  dtype=object, name=values.name)
            indexes.append(values)
        data.index = (indexes[0] if data.index.nlevels == 1 else
                      pd.MultiIndex.from_arrays(indexes, names=data.index.names))
    return data


def encode_figure_table(data: pd.DataFrame) -> bytes:
    """Public lossless table codec for full analytical resources in a bundle."""
    if not isinstance(data, pd.DataFrame):
        raise TypeError("A figure table must be a DataFrame.")
    return _table_parquet_bytes(data)


def decode_figure_table(payload: bytes) -> pd.DataFrame:
    """Restore native numerical columns and any schema-tagged object scalars."""
    return _table_from_parquet(payload)
