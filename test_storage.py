import io
from unittest.mock import MagicMock

import storage


def test_save_file_writes_locally_without_s3(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.delenv("S3_BUCKET", raising=False)
    storage.save_file(io.BytesIO(b"hello"), "a.txt")
    assert (tmp_path / "a.txt").read_bytes() == b"hello"


def test_save_file_uploads_to_s3_when_bucket_is_set(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    fake_client = MagicMock()
    monkeypatch.setattr(storage.boto3, "client", lambda name: fake_client)
    storage.save_file(io.BytesIO(b"hello"), "a.txt")
    fake_client.upload_file.assert_called_once_with(str(tmp_path / "a.txt"), "my-bucket", "a.txt")


def test_get_local_path_downloads_from_s3_when_file_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(storage, "UPLOAD_DIR", str(tmp_path))
    monkeypatch.setenv("S3_BUCKET", "my-bucket")
    fake_client = MagicMock()
    monkeypatch.setattr(storage.boto3, "client", lambda name: fake_client)
    path = storage.get_local_path("a.txt")
    fake_client.download_file.assert_called_once_with("my-bucket", "a.txt", str(tmp_path / "a.txt"))
    assert path == str(tmp_path / "a.txt")