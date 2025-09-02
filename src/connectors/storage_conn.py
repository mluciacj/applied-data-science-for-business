# src/data_to_decisions/connectors/s3.py
from __future__ import annotations

from typing import Iterable, Optional, Any, List
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
import os

# Ensure dependency is present
try:
    import s3fs  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError("S3Reader requires s3fs. Install with: pip install s3fs boto3") from e


class S3Reader:
    """
    Minimal S3/MinIO reader using pandas + s3fs.

    Auth comes from the normal AWS chain (env vars, profiles, etc.).
    For MinIO, pass endpoint_url (e.g., 'http://localhost:9000').

    Usage:
        s3 = S3Reader()  # AWS
        df = s3.read_df("s3://my-bucket/path/data.parquet")

        s3 = S3Reader(endpoint_url="http://localhost:9000")  # MinIO
        df = s3.read_df(["s3://data/2025/08/a.csv", "s3://data/2025/08/b.csv"], fmt="csv")
    """

    def __init__(self,
                endpoint_url: str | None = None,
                anon: bool = False,
                use_ssl: bool = True,
                profile: str | None = None,
                token: str | None = None,
                requester_pays: bool = False):

        self.storage_options: dict[str, Any] = {"anon": anon}
        region_name = os.getenv("S3_REGION")
        secret = os.getenv("AWS_SECRET_ACCESS_KEY")
        key = os.getenv("AWS_ACCESS_KEY_ID")

        if endpoint_url is not None:
            self.storage_options["client_kwargs"] = {"endpoint_url": endpoint_url}
            self.storage_options["use_ssl"] = use_ssl
        if region_name:
            self.storage_options.setdefault("client_kwargs", {})["region_name"] = region_name
        if profile:
            self.storage_options["profile"] = profile           # for AWS profiles/SSO
        if key and secret:
            self.storage_options.update({"key": key, "secret": secret})
        if token:
                self.storage_options["token"] = token
        if requester_pays:
            self.storage_options["requester_pays"] = True

    def read_df(self, path: str | Iterable[str], fmt: Optional[str] = "csv", **kwargs: Any) -> pd.DataFrame:
        files = [path] if isinstance(path, str) else list(path)
        if not files:
            return pd.DataFrame()

        fmt = (fmt or self._infer_format(files[0])).lower()
        if fmt not in {"csv", "parquet"}:
            raise ValueError("Unsupported fmt. Use 'csv' or 'parquet'.")

        reader = pd.read_csv if fmt == "csv" else pd.read_parquet

        frames: List[pd.DataFrame] = []
        for f in files:
            df = reader(f, storage_options=self.storage_options, **kwargs)
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    @staticmethod
    def _infer_format(path: str) -> str:
        p = path.lower()
        if p.endswith(".csv"):
            return "csv"
        if p.endswith(".parquet") or p.endswith(".pq"):
            return "parquet"
        return "unknown"  # default
