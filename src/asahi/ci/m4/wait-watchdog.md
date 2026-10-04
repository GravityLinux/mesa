# HK long-wait diagnostics for test builds

Configure **both native and guest Mesa** with:

```sh
meson configure BUILD -Dhk-long-wait-watchdog=true
```

This option defaults off. An enabled build always monitors waits, including in
release builds; no `HK_DEBUG` switch is needed. It does not enable command-record
tracing, GPU tracing, synchronous submission, or extra GPU queries.

After a driver wait remains active for five seconds, an independent thread emits
one report for that wait scope to stderr and
`/tmp/hk-wait-watchdog-PID.log`. `HK_WAIT_WATCHDOG_FILE` overrides the filename;
a literal `%p` is replaced by the process ID. The file is appended with mode 0600,
symlinks are refused, and the file is written before stderr. Use a writable
persistent directory when the guest's `/tmp` will disappear on teardown.

The original wait arguments, timeout, result and errno are preserved. A
permanently blocked call is diagnosed without requiring it to return. Nested
wait layers can each report once: for example, a WSI scope provides the Vulkan
semaphore handles and its nested DRM scope supplies actual syncobj handles.
Zero-time polls and waits with less than five seconds remaining aren't registered.

## What the report contains

* Monotonic timestamp/age, process and thread IDs, wait kind, fd namespace,
  waited objects, points, flags and absolute deadline (where applicable).
* The last 16 submission attempts for each live HK kernel queue, including empty
  submissions implemented by syncobj transfers. Records distinguish an ioctl
  still executing from a returned submission and preserve result/errno.
* Submission input and output handles/points (up to 64 of each, explicitly marked
  if truncated), plus the private queue progress handle and point.
* Whether the driver has observed the private queue timeline complete through
  an existing successful wait or completed-point query. Availability waits and
  `LAST_SUBMITTED` queries do **not** imply completion. Neither do arbitrary
  application semaphore signals. Reset/import/destroy invalidates future
  matching against that handle. Binary completion isn't inferred from a handle
  that can be recycled.
* The latest 64 syncobj/virtio ioctl or host response returns. Errno is meaningful
  only for calls that report failure. A virtio host-sequence wait also includes
  the current shared-memory host sequence number.

`completion=not-observed` means exactly that: it is **not proof of unfinished
GPU work**. Normal asynchronous execution may never query/wait on the private
queue timeline. Correlate with the host kernel's submission/completion and
backpressure diagnostics to locate a lost signal across guest/VMM/kernel. The
dump deliberately does not delay itself by querying the GPU or virtio host.
This cannot observe a game-only futex wait that never enters Mesa.

Covered wait paths are common Vulkan fence/semaphore/timeline sync waits, DRM
syncobj waits, WSI acquire/present/present-wait, HK submit ioctls, virtgpu ioctls
and sync-file fences, and the virtio shared-memory host-response wait.

## Cost and lifetime

There is no per-record I/O before a report, and no diagnostic GPU querying.
There **is** bounded in-memory bookkeeping: clocks, a mutex around registered
long waits and histories, and 16 fixed-sized records per live queue (about
35 KiB/queue). The watcher sleeps until a registered wait's five-second
threshold. These costs are included in the watchdog-build baseline; “zero cost”
would not accurately describe a watchdog retaining submission history.

Queue destruction releases its history. Fork children start a fresh watcher
without inherited parent waits. Teardown cancels blocked diagnostic output,
so a full stderr pipe cannot hold up driver unloading. The file sink is used
first, but unwritable files or storage failures can still prevent persistence.

## Host validation

With `-Dbuild-tests=true`, run the `wait watchdog` Meson test. The fixture blocks
on a pipe indefinitely; the controller verifies the dump at five seconds
**before** releasing the pipe. It verifies once-per-scope reporting, identical
stderr/file output, history wrapping, failed submissions, retired handles,
short/finished waits, fork with a sleeping watcher, full 64-bit object IDs in
32-bit builds, and teardown with an undrained stderr pipe. No GPU is needed.
