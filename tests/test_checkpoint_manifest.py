from pathlib import Path

from src.engine_v2.checkpoint_v2 import AsyncCheckpointManagerV2


def test_manifest_hash_is_order_independent_and_content_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "shard_000.bin"
    second = tmp_path / "shard_001.bin"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    expected = AsyncCheckpointManagerV2.calculate_manifest_sha256([first, second])
    reordered = AsyncCheckpointManagerV2.calculate_manifest_sha256([second, first])
    assert reordered == expected

    second.write_bytes(b"changed")
    changed = AsyncCheckpointManagerV2.calculate_manifest_sha256([first, second])
    assert changed != expected
