"""Read videos out of the FineVideo Parquet files.

Each row of a Parquet file is one video; its `mp4` column holds the whole
mp4 file as bytes. A video's id is "<file name>:<row number>", e.g.
"train-00003-of-01357.parquet:4" (section 4 of the brief).

A FineVideo file stores all its rows in ONE row group, so Parquet cannot hand
out a single row without reading the whole 0.5 GB column. That is why one
reader reads each file once and passes the bytes on to the decode workers,
instead of every worker opening the file for the rows it wants.
"""

from pathlib import Path

import pyarrow.parquet as pq


def list_parquet_files(folder):
    files = sorted(Path(folder).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files found in {folder}")
    return files


def video_id(parquet_path, row):
    return f"{Path(parquet_path).name}:{row}"


def iter_videos(parquet_path):
    """Yield (video_id, mp4_bytes_or_None, error_or_None) for every row.

    Never raises for a damaged file: problems come back as an error string,
    so one bad file cannot stop the run.
    """
    try:
        table = pq.read_table(parquet_path, columns=["mp4"])
    except Exception as exc:  # unreadable or truncated file
        yield from _rows_of_unreadable_file(parquet_path, exc)
        return

    column = table.column("mp4")
    for row in range(len(column)):
        vid = video_id(parquet_path, row)
        try:
            data = column[row].as_py()
        except Exception as exc:
            yield vid, None, f"could not read row: {exc}"
            continue
        if not data:
            yield vid, None, "empty mp4 column"
        else:
            yield vid, data, None


def _rows_of_unreadable_file(parquet_path, exc):
    """Report every row of a file whose mp4 column could not be read.

    If the footer is still readable we know how many rows there were and
    give each its own error line. If not, there is no row count to go on,
    so the file gets a single error line under row 0.
    """
    message = f"could not read parquet file: {type(exc).__name__}"
    try:
        n_rows = pq.ParquetFile(parquet_path).metadata.num_rows
    except Exception:
        n_rows = 1
    for row in range(max(n_rows, 1)):
        yield video_id(parquet_path, row), None, message
