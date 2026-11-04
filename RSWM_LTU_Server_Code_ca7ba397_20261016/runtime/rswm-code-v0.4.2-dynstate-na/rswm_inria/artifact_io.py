"""Complete, exclusive PNG publication for cooperative materialization writers.

This prevents partial final filenames and overwriting an existing artifact. It
does not attest the filesystem or prevent another actor modifying a published
file later; independent retained-hash and pixel audits remain necessary.
"""
import io
import os
from pathlib import Path
import uuid

from PIL import Image


class _CompleteWriteStream:
    """Give streaming serializers the same short-write check as byte writers."""
    def __init__(self, stream): self.stream = stream
    def __getattr__(self, name): return getattr(self.stream, name)
    def write(self, content):
        written = self.stream.write(content)
        if written != len(content):
            raise OSError("Incomplete artifact write")
        return written


def save_stream_exclusive(path, writer):
    """Stream, synchronize and exclusively publish without buffering a model."""
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    owned = False
    try:
        with temporary.open("xb") as stream:
            owned = True
            writer(_CompleteWriteStream(stream))
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes the complete inode and fails if the name is occupied.
        os.link(temporary, path)
        temporary.unlink()
        owned = False
        if os.name == "posix":
            descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if owned:
            temporary.unlink(missing_ok=True)


def save_bytes_exclusive(path, content):
    """Publish complete synchronized bytes using the existing exclusive primitive."""
    save_stream_exclusive(path, lambda stream: stream.write(content))


def save_png_exclusive(path, pixels, **options):
    # Keep incomplete encoding out of the filesystem, including staging names.
    encoded = io.BytesIO()
    Image.fromarray(pixels).save(encoded, format="PNG", **options)
    save_bytes_exclusive(path, encoded.getbuffer())
