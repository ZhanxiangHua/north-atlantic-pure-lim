"""Publish and restore lossless monthly weight differences against a named base checkpoint."""
import argparse
import hashlib
import io
import json
import os
import struct
import tempfile
import zipfile
import zlib
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def checksum(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def temporary_output(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".checkpoint-", dir=path.parent)
    os.close(handle)
    temporary = Path(temporary)
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def json_bytes(data):
    return (json.dumps(data, sort_keys=True, indent=2) + "\n").encode()


def sections(blob):
    """Locate uncompressed NPY payloads; preserve exact original ZIP headers and directory bytes."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        infos = sorted(archive.infolist(), key=lambda z: z.header_offset)
        for i in infos:
            if i.compress_type != zipfile.ZIP_STORED or i.flag_bits & 1:
                raise ValueError("Use uncompressed numeric NPZ checkpoints")
            off = i.header_offset
            head = struct.unpack_from("<IHHHHHIIIHH", blob, off)
            if head[0] != 0x04034B50:
                raise ValueError("Invalid local ZIP header")
            start = off + 30 + head[-2] + head[-1]
            yield i.filename, start, start + i.file_size


def encode_month(base, source, path):
    data = {"format": 1, "files": {}}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:

        def store(name, payload):
            entry = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o644 << 16
            output.writestr(entry, payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)

        store("model.json", (source / "model.json").read_bytes())
        for fn in ("representation.npz", "parameters.npz"):
            blob = (source / fn).read_bytes()
            entries = []
            last = 0
            info = dict(
                base_sha256=checksum(base / fn),
                target_sha256=checksum(source / fn),
                entries=entries,
            )
            with zipfile.ZipFile(base / fn) as old:
                names = set(old.namelist())
                for k, (key, start, end) in enumerate(sections(blob)):
                    stem = fn + "/" + str(k)
                    store(stem + ".gap", blob[last:start])
                    last = end
                    target = blob[start:end]
                    previous = old.read(key) if key in names else None
                    entry = dict(key=key, stem=stem)
                    if previous == target:
                        entry["mode"] = "copy"
                    else:
                        arr = np.load(io.BytesIO(target), allow_pickle=False)
                        ref = (
                            np.load(io.BytesIO(previous), allow_pickle=False)
                            if previous is not None
                            else None
                        )
                        if arr.dtype.hasobject:
                            raise ValueError("Object arrays are invalid in frozen weights")
                        offset = len(target) - arr.nbytes
                        order = (
                            "F" if arr.flags.f_contiguous and not arr.flags.c_contiguous else "C"
                        )
                        entry.update(order=order, mode="full")
                        store(stem + ".npyheader", target[:offset])
                        new = target[offset:]
                        if ref is not None and arr.shape == ref.shape and arr.dtype == ref.dtype:
                            ref_bytes = ref.tobytes(order=order)
                            if ref_bytes == new:
                                entry["mode"] = "reorder"
                            else:
                                change = (
                                    np.frombuffer(ref_bytes, dtype=np.uint8)
                                    ^ np.frombuffer(new, dtype=np.uint8)
                                ).tobytes()
                                if len(zlib.compress(change, 3)) < 0.96 * len(
                                    zlib.compress(new, 3)
                                ):
                                    entry["mode"] = "xor"
                                    new = change
                        if entry["mode"] != "reorder":
                            store(stem + ".data", new)
                    entries.append(entry)
            store(fn + "/tail", blob[last:])
            data["files"][fn] = info
        store("layout.json", json_bytes(data))


def build(base, source, out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = dict(format=1, base="north-atlantic-pure-lim v1.0.0", months={})
    for month in range(1, 13):
        m = f"{month:02}"
        pack = out / (m + ".delta.zip")
        with temporary_output(pack) as tmp:
            encode_month(base / m, source / m, tmp)
        manifest["months"][m] = dict(
            file=pack.name,
            sha256=checksum(pack),
            bytes=pack.stat().st_size,
            base_id=json.loads((base / m / "model.json").read_text())["id"],
            model_id=json.loads((source / m / "model.json").read_text())["id"],
        )
    with temporary_output(out / "manifest.json") as tmp:
        tmp.write_bytes(json_bytes(manifest))
    return manifest


def restore(base, delta, out):
    if base.resolve() == out.resolve():
        raise ValueError("Choose a separate destination for the research checkpoint")
    info = json.loads((delta / "manifest.json").read_text())
    if info.get("format") != 1:
        raise ValueError("Unknown checkpoint difference format")
    for m in [f"{i:02}" for i in range(1, 13)]:
        row = info["months"][m]
        if json.loads((base / m / "model.json").read_text())["id"] != row["base_id"]:
            raise ValueError(f"Different baseline weights in month {m}")
        pack = delta / row["file"]
        if checksum(pack) != row["sha256"]:
            raise ValueError(f"Damaged difference archive in month {m}")
        with zipfile.ZipFile(pack) as z:
            target = json.loads(z.read("model.json"))
            if target["id"] != row["model_id"]:
                raise ValueError("Invalid target identifier")
            for fn, meta in json.loads(z.read("layout.json"))["files"].items():
                if (
                    fn not in ("parameters.npz", "representation.npz")
                    or checksum(base / m / fn) != meta["base_sha256"]
                ):
                    raise ValueError(f"Different baseline file {m}/{fn}")
                with temporary_output(out / m / fn) as tmp, zipfile.ZipFile(base / m / fn) as old:
                    with tmp.open("wb") as stream:
                        for e in meta["entries"]:
                            stem = e["stem"]
                            mode = e["mode"]
                            stream.write(z.read(stem + ".gap"))
                            if mode == "copy":
                                stream.write(old.read(e["key"]))
                                continue
                            stream.write(z.read(stem + ".npyheader"))
                            if mode == "full":
                                stream.write(z.read(stem + ".data"))
                                continue
                            arr = np.load(io.BytesIO(old.read(e["key"])), allow_pickle=False)
                            payload = arr.tobytes(order=e["order"])
                            if mode == "xor":
                                payload = (
                                    np.frombuffer(payload, dtype=np.uint8)
                                    ^ np.frombuffer(z.read(stem + ".data"), dtype=np.uint8)
                                ).tobytes()
                            elif mode != "reorder":
                                raise ValueError("Unknown numeric-array recovery method")
                            stream.write(payload)
                        stream.write(z.read(fn + "/tail"))
                    if (
                        checksum(tmp) != target["files"][fn]
                        or target["files"][fn] != meta["target_sha256"]
                    ):
                        raise ValueError(f"Restored checksum mismatch in {m}/{fn}")
            with temporary_output(out / m / "model.json") as tmp:
                tmp.write_bytes(z.read("model.json"))
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="operation", required=True)
    for command in ("build", "restore"):
        q = sub.add_parser(command)
        q.add_argument("--base", required=True, type=Path)
        q.add_argument("--source" if command == "build" else "--delta", required=True, type=Path)
        q.add_argument("--output", required=True, type=Path)
    a = p.parse_args()
    if a.operation == "build":
        build(a.base, a.source, a.output)
    else:
        restore(a.base, a.delta, a.output)
