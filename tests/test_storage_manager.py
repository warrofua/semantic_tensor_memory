from pathlib import Path
import time
import os

from semantic_tensor_analysis.storage.manager import StorageManager


def _touch(file: Path, *, days_ago: int = 0) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_bytes(b"test")
    if days_ago:
        past = time.time() - days_ago * 86400
        # Manually set mtime/atime for Python < 3.11 compatibility
        os.utime(file, (past, past))


def test_storage_stats_and_cleanup(tmp_path):
    storage_dir = tmp_path / "universal"
    old_file = storage_dir / "session_old.pkl"
    new_file = storage_dir / "session_new.pkl"
    _touch(old_file, days_ago=10)
    _touch(new_file, days_ago=1)

    manager = StorageManager(storage_dir)
    stats = manager.get_stats()

    assert stats.total_files == 2
    assert stats.total_size_bytes >= 2
    assert stats.oldest_mtime is not None
    assert stats.newest_mtime is not None

    # Dry-run cleanup should not delete
    removed, freed = manager.cleanup_old_sessions(days=5, dry_run=True)
    assert removed == 1
    assert freed >= 1
    assert old_file.exists()

    # Actual cleanup
    removed, freed = manager.cleanup_old_sessions(days=5, dry_run=False)
    assert removed == 1
    assert freed >= 1
    assert not old_file.exists()
    assert new_file.exists()


def test_archive_sessions(tmp_path):
    storage_dir = tmp_path / "src"
    archive_dir = tmp_path / "archive"
    files = [storage_dir / f"session_{i}.pkl" for i in range(3)]
    for f in files:
        _touch(f)

    manager = StorageManager(storage_dir)
    moved = manager.archive_sessions(archive_dir)
    assert moved == 3
    for f in files:
        assert not f.exists()
        assert (archive_dir / f.name).exists()
