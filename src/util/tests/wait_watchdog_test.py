#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
import os
import pathlib
import subprocess
import sys
import tempfile
import time

binary = sys.argv[1]
with tempfile.TemporaryDirectory(prefix="mesa-wait-watchdog-") as directory:
    directory = pathlib.Path(directory)
    def start(mode):
        env = dict(os.environ, HK_WAIT_WATCHDOG_FILE=str(directory / (mode + "-%p.log")))
        return subprocess.Popen([binary, mode], env=env, stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # These run concurrently; each test must actually cross the five-second
    # boundary. Output is read only after the wait has been diagnosed.
    runs = {mode: start(mode) for mode in ("blocked", "short", "fork", "full-stderr")}
    start_time = time.monotonic()
    try:
        while time.monotonic() - start_time < 8:
            files = list(directory.glob("*.log"))
            if len(files) >= 3 and all("END HK LONG WAIT" in f.read_text() for f in files):
                break
            time.sleep(0.05)
        elapsed = time.monotonic() - start_time
        assert 4.8 <= elapsed < 8, elapsed
        assert not list(directory.glob("short-*.log"))
        for mode in ("blocked", "fork", "full-stderr"):
            process = runs[mode]
            assert process.poll() is None, f"{mode}: dump only happened after return"
            files = list(directory.glob(mode + "-*.log"))
            assert len(files) == 1, (mode, files)
            text = files[0].read_text()
            assert text.count("HK LONG WAIT pid=") == 1, text
            assert "tid=" in text and "fd=42" in text and "point=20" in text
            assert "object=0xabcdef0123456789" in text
            assert "object=0x4d point=20" in text and "object=0x58 point=33" in text
            if mode == "blocked":
                assert "queue=0x7 last=16" in text
                assert " submit=4 " not in text and " submit=5 " in text
                assert " submit=20 " in text and "completion=observed" in text
                assert "queue=0x9 last=1" in text
                assert "result=-1 errno=16 completion=not-observed" in text
                retired = text.split("queue=0xa last=1", 1)[1].split("QUEUE", 1)[0]
                assert "completion=not-observed" in retired
                assert "test-ioctl ret=-1 errno=16" in text
        # Verify no second report for the same still-blocked wait.
        time.sleep(1.2)
        for f in directory.glob("*.log"):
            assert f.read_text().count("HK LONG WAIT pid=") == 1
        for mode, process in runs.items():
            if mode == "full-stderr":
                # Don't drain stderr until after process termination. The
                # destructor must cancel diagnostic I/O rather than hang.
                process.stdin.write(b"x")
                process.stdin.flush()
                process.wait(timeout=3)
                process.communicate(timeout=3)
            else:
                stdout, stderr = process.communicate(b"x", timeout=3)
                if mode != "short":
                    text = next(directory.glob(mode + "-*.log")).read_bytes()
                    assert stderr == text, (mode, len(stderr), len(text))
                else:
                    assert stderr == b"", stderr
            assert process.returncode == 0, (mode, process.returncode)
        print(f"PASS: blocked wait diagnosed after {elapsed:.2f}s before return; "
              "one report, dual sinks, history, failure/retirement, short waits, "
              "fork and blocked stderr teardown")
    finally:
        for process in runs.values():
            if process.poll() is None:
                process.kill()
                process.wait()
