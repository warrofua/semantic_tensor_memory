"""
Lightweight storage management utilities.

Provides basic stats and cleanup helpers for the universal session store without
affecting core app behavior. Designed to operate safely by default (no deletions
unless explicitly requested).
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_STORAGE_DIR = Path("data/universal")


@dataclass
class StorageStats:
    total_files: int
    total_size_bytes: int
    oldest_mtime: Optional[datetime]
    newest_mtime: Optional[datetime]
    sample_files: List[Path]

    @property
    def total_size_mb(self) -> float:
        return self.total_size_bytes / 1024 / 1024


class StorageManager:
    """Manage lifecycle of persisted session files."""

    def __init__(self, storage_dir: Path | str = DEFAULT_STORAGE_DIR):
        self.storage_dir = Path(storage_dir)

    def _iter_session_files(self) -> Iterable[Path]:
        if not self.storage_dir.exists():
            return []
        return sorted(self.storage_dir.glob("session_*.pkl"))

    def get_stats(self) -> StorageStats:
        """Return basic storage stats (count, size, age)."""
        files = list(self._iter_session_files())
        total_size = sum(f.stat().st_size for f in files)
        mtimes = [datetime.fromtimestamp(f.stat().st_mtime) for f in files]
        oldest = min(mtimes) if mtimes else None
        newest = max(mtimes) if mtimes else None
        sample = files[:5]
        return StorageStats(
            total_files=len(files),
            total_size_bytes=total_size,
            oldest_mtime=oldest,
            newest_mtime=newest,
            sample_files=sample,
        )

    def cleanup_old_sessions(self, days: int, dry_run: bool = True) -> Tuple[int, int]:
        """
        Remove sessions older than N days.

        Returns (removed_count, freed_bytes). With dry_run=True, performs no deletions.
        """
        cutoff = datetime.now() - timedelta(days=days)
        removed = 0
        freed = 0
        for f in self._iter_session_files():
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            if mtime < cutoff:
                freed += f.stat().st_size
                removed += 1
                if not dry_run:
                    try:
                        f.unlink()
                    except Exception:
                        pass
        return removed, freed

    def archive_sessions(self, target_dir: Path | str, pattern: str = "session_*.pkl") -> int:
        """
        Move matching session files to target directory.

        Returns number of files moved.
        """
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        moved = 0
        for f in self.storage_dir.glob(pattern):
            shutil.move(str(f), target / f.name)
            moved += 1
        return moved


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Storage management for STA sessions")
    parser.add_argument("--storage-dir", default=str(DEFAULT_STORAGE_DIR), help="Path to session storage directory")
    parser.add_argument("--stats", action="store_true", help="Show storage stats")
    parser.add_argument("--cleanup-days", type=int, help="Delete sessions older than N days")
    parser.add_argument("--apply", action="store_true", help="Perform deletions (without this flag, dry-run)")
    parser.add_argument("--archive-to", type=str, help="Move session_*.pkl files to target directory")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manager = StorageManager(args.storage_dir)

    if args.stats:
        stats = manager.get_stats()
        print(f"Files: {stats.total_files}")
        print(f"Size: {stats.total_size_mb:.2f} MB")
        print(f"Oldest: {stats.oldest_mtime}")
        print(f"Newest: {stats.newest_mtime}")
        if stats.sample_files:
            print("Sample files:")
            for f in stats.sample_files:
                print(f"  - {f.name}")

    if args.cleanup_days is not None:
        removed, freed = manager.cleanup_old_sessions(args.cleanup_days, dry_run=not args.apply)
        mode = "Dry-run" if not args.apply else "Deleted"
        print(f"{mode}: {removed} files (~{freed/1024/1024:.2f} MB)")

    if args.archive_to:
        moved = manager.archive_sessions(args.archive_to)
        print(f"Moved {moved} files to {args.archive_to}")


if __name__ == "__main__":
    main()
