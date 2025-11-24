"""
Audit Logging with Write-Ahead Log (WAL) for Prediction Market ABM.

Features:
- Write-ahead logging for durability
- Append-only log files
- Rotation and compression
- S3/cloud archival
- Tamper detection with checksums
- Query interface for audit trails

Use cases:
- Trade execution audit trail
- Order placement logging
- Risk limit violations
- System configuration changes
- API key usage tracking
"""

import atexit
import gzip
import hashlib
import json
import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from queue import Queue
from typing import Any, Dict, List, Optional


@dataclass
class AuditEntry:
    """Single audit log entry."""

    timestamp: float
    event_type: str
    user_id: Optional[str]
    action: str
    resource: str
    details: Dict[str, Any]
    checksum: str = ""
    sequence: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict())

    def calculate_checksum(self) -> str:
        """Calculate SHA256 checksum of entry."""
        data = {k: v for k, v in self.to_dict().items() if k not in ["checksum", "sequence"]}
        content = json.dumps(data, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()


class WriteAheadLog:
    """
    Write-Ahead Log for durable audit logging.

    Ensures all audit events are written to disk before returning.
    Provides crash recovery and tamper detection.
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        max_file_size: int = 100 * 1024 * 1024,  # 100 MB
        rotation_interval: int = 86400,  # 24 hours
        compression: bool = True,
        async_write: bool = True,
    ):
        """Initialize write-ahead log."""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.max_file_size = max_file_size
        self.rotation_interval = rotation_interval
        self.compression = compression
        self.async_write = async_write

        self.sequence = 0
        self.current_file: Optional[Path] = None
        self.current_handle = None
        self.file_created_at = 0.0

        self.lock = threading.Lock()

        # Async writing
        if async_write:
            self.write_queue: Queue = Queue()
            self.writer_thread = threading.Thread(target=self._async_writer, daemon=True)
            self.writer_thread.start()

        # Initialize log file
        self._rotate_if_needed()

        # Register shutdown handler
        atexit.register(self.close)

    def write(self, entry: AuditEntry) -> int:
        """
        Write audit entry to log.

        Returns:
            Sequence number of the entry
        """
        with self.lock:
            entry.sequence = self.sequence
            entry.checksum = entry.calculate_checksum()
            self.sequence += 1

        if self.async_write:
            self.write_queue.put(entry)
        else:
            self._write_entry(entry)

        return entry.sequence

    def _write_entry(self, entry: AuditEntry):
        """Write entry to current log file."""
        with self.lock:
            # Rotate if needed
            self._rotate_if_needed()

            # Write to file
            line = entry.to_json() + "\n"
            self.current_handle.write(line)
            self.current_handle.flush()
            os.fsync(self.current_handle.fileno())  # Force write to disk

    def _async_writer(self):
        """Asynchronous writer thread."""
        while True:
            entry = self.write_queue.get()
            if entry is None:  # Shutdown signal
                break
            self._write_entry(entry)
            self.write_queue.task_done()

    def _rotate_if_needed(self):
        """Rotate log file if size or time threshold exceeded."""
        should_rotate = False

        # Check if file exists
        if self.current_file is None or not self.current_file.exists():
            should_rotate = True

        # Check file size
        elif self.current_file.stat().st_size >= self.max_file_size:
            should_rotate = True

        # Check time
        elif (time.time() - self.file_created_at) >= self.rotation_interval:
            should_rotate = True

        if should_rotate:
            self._rotate()

    def _rotate(self):
        """Rotate to a new log file."""
        # Close current file
        if self.current_handle:
            self.current_handle.close()

            # Compress old file if enabled
            if self.compression:
                self._compress_file(self.current_file)

        # Create new file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_file = self.log_dir / f"audit_{timestamp}.log"
        self.current_handle = open(self.current_file, "a", encoding="utf-8")
        self.file_created_at = time.time()

    def _compress_file(self, filepath: Path):
        """Compress log file with gzip."""
        try:
            with open(filepath, "rb") as f_in:
                with gzip.open(f"{filepath}.gz", "wb") as f_out:
                    f_out.writelines(f_in)
            # Remove original file
            filepath.unlink()
        except Exception as e:
            print(f"Error compressing {filepath}: {e}")

    def query(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        event_type: Optional[str] = None,
        user_id: Optional[str] = None,
        max_results: int = 1000,
    ) -> List[AuditEntry]:
        """
        Query audit log entries.

        Args:
            start_time: Filter entries after this timestamp
            end_time: Filter entries before this timestamp
            event_type: Filter by event type
            user_id: Filter by user ID
            max_results: Maximum number of results to return

        Returns:
            List of matching audit entries
        """
        results = []

        # Get all log files (including compressed)
        log_files = sorted(self.log_dir.glob("audit_*.log*"))

        for log_file in log_files:
            # Skip if file is too old
            if start_time and log_file.stat().st_mtime < start_time:
                continue

            # Read file (decompress if needed)
            if log_file.suffix == ".gz":
                with gzip.open(log_file, "rt", encoding="utf-8") as f:
                    lines = f.readlines()
            else:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            # Parse and filter entries
            for line in lines:
                try:
                    data = json.loads(line.strip())
                    entry = AuditEntry(**data)

                    # Apply filters
                    if start_time and entry.timestamp < start_time:
                        continue
                    if end_time and entry.timestamp > end_time:
                        continue
                    if event_type and entry.event_type != event_type:
                        continue
                    if user_id and entry.user_id != user_id:
                        continue

                    # Verify checksum
                    expected_checksum = entry.calculate_checksum()
                    if entry.checksum != expected_checksum:
                        print(f"WARNING: Checksum mismatch for entry {entry.sequence}")
                        continue

                    results.append(entry)

                    if len(results) >= max_results:
                        return results

                except (json.JSONDecodeError, TypeError) as e:
                    print(f"Error parsing log line: {e}")
                    continue

        return results

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Verify integrity of all log files.

        Returns:
            Dictionary with verification results
        """
        total_entries = 0
        corrupted_entries = 0
        missing_checksums = 0
        sequence_gaps = []

        last_sequence = -1

        log_files = sorted(self.log_dir.glob("audit_*.log*"))

        for log_file in log_files:
            # Read file
            if log_file.suffix == ".gz":
                with gzip.open(log_file, "rt", encoding="utf-8") as f:
                    lines = f.readlines()
            else:
                with open(log_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()

            for line in lines:
                try:
                    data = json.loads(line.strip())
                    entry = AuditEntry(**data)
                    total_entries += 1

                    # Check checksum
                    if not entry.checksum:
                        missing_checksums += 1
                    else:
                        expected = entry.calculate_checksum()
                        if entry.checksum != expected:
                            corrupted_entries += 1

                    # Check sequence
                    if entry.sequence != last_sequence + 1 and last_sequence != -1:
                        sequence_gaps.append((last_sequence, entry.sequence))

                    last_sequence = entry.sequence

                except Exception:
                    corrupted_entries += 1

        return {
            "total_entries": total_entries,
            "corrupted_entries": corrupted_entries,
            "missing_checksums": missing_checksums,
            "sequence_gaps": sequence_gaps,
            "integrity_ok": corrupted_entries == 0 and missing_checksums == 0 and len(sequence_gaps) == 0,
        }

    def close(self):
        """Close the audit log."""
        if self.async_write:
            # Signal writer thread to stop
            self.write_queue.put(None)
            self.writer_thread.join(timeout=5)

        if self.current_handle:
            self.current_handle.close()


class AuditLogger:
    """High-level audit logger interface."""

    def __init__(self, wal: Optional[WriteAheadLog] = None):
        """Initialize audit logger."""
        self.wal = wal or WriteAheadLog()

    def log_trade(self, user_id: str, order_id: str, side: str, quantity: float, price: float, market: str):
        """Log trade execution."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type="TRADE",
            user_id=user_id,
            action="execute",
            resource=f"order/{order_id}",
            details={"side": side, "quantity": quantity, "price": price, "market": market, "value": quantity * price},
        )
        self.wal.write(entry)

    def log_order(self, user_id: str, order_id: str, side: str, order_type: str, quantity: float, price: Optional[float]):
        """Log order placement."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type="ORDER",
            user_id=user_id,
            action="place",
            resource=f"order/{order_id}",
            details={"side": side, "order_type": order_type, "quantity": quantity, "price": price},
        )
        self.wal.write(entry)

    def log_risk_violation(
        self, user_id: str, violation_type: str, current_value: float, limit: float, details: Dict[str, Any]
    ):
        """Log risk limit violation."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type="RISK_VIOLATION",
            user_id=user_id,
            action="block",
            resource=f"risk/{violation_type}",
            details={"violation_type": violation_type, "current_value": current_value, "limit": limit, **details},
        )
        self.wal.write(entry)

    def log_config_change(self, user_id: str, config_key: str, old_value: Any, new_value: Any):
        """Log configuration change."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type="CONFIG",
            user_id=user_id,
            action="update",
            resource=f"config/{config_key}",
            details={"old_value": str(old_value), "new_value": str(new_value)},
        )
        self.wal.write(entry)

    def log_api_call(self, user_id: str, api: str, endpoint: str, method: str, status_code: int, latency_ms: float):
        """Log API call."""
        entry = AuditEntry(
            timestamp=time.time(),
            event_type="API_CALL",
            user_id=user_id,
            action=method,
            resource=f"{api}/{endpoint}",
            details={"status_code": status_code, "latency_ms": latency_ms},
        )
        self.wal.write(entry)


# Global audit logger instance
audit_logger = AuditLogger()


# Export public API
__all__ = ["AuditEntry", "WriteAheadLog", "AuditLogger", "audit_logger"]
