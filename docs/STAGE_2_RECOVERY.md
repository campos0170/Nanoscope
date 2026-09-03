# Stage 2 crash-recovery contract

Stage 2 protects unsaved analysis if NanoScope or the computer stops before the user can
save a project. It is a recovery mechanism, not a second project format and not a promise
that every keystroke has already reached disk.

## User-visible behaviour

1. NanoScope takes recovery snapshots while a session has unsaved work. The first snapshot
   is due after roughly 15 seconds without a new edit. Continued editing cannot postpone a
   snapshot forever: the oldest unsaved revision is checkpointed within 60 seconds.
2. Recovery files live in NanoScope's private application-data directory. They never replace
   a user-named `.nanoscope` project and are not added to **Open Recent Project**.
3. On a later launch, NanoScope offers each recoverable session with three explicit choices:
   **Recover**, **Discard**, and **Show in Finder**. Showing the files is non-destructive and
   returns to the same decision; closing or cancelling the prompt keeps the recovery data.
4. **Recover** opens the normal `ProjectState` payload through the normal restore path. A
   recovered session is visibly marked **Unsaved** (`•` in the main title and the same state
   in Gallery/Inspector) until the user explicitly saves it.
5. **Discard** removes both retained generations for only the session named by the prompt.
   Other recoverable sessions remain available.
6. If the newest generation is incomplete or corrupt, NanoScope validates it and falls back
   to the previous valid generation. A bad newest write therefore cannot hide the last good
   checkpoint.

Recovery is deliberately conservative. An explicit project save retires only recovery
revisions represented by the state that was actually committed. If an edit happens while a
save is gathering or writing, the newer recovery revision remains available. A cancelled or
failed save retires nothing.

## Timing and snapshot boundary

`MainWindow` owns a monotonically increasing `session_revision`. Every persistent edit that
marks the project dirty also advances that revision. The GUI thread gathers one immutable
`ProjectState` and attaches the revision before handing it to the background serializer;
the worker never reads live Qt widgets, `Session`, or a mutating pandas frame.

The controller uses two deadlines:

| Deadline | Meaning |
| --- | --- |
| 15 seconds idle | Checkpoint after editing settles. A later edit moves this deadline. |
| 60 seconds maximum | Checkpoint even if edits keep arriving. A later edit does not move this deadline. |

Only one recovery write runs at a time. Edits that arrive during a write are coalesced into
one latest pending immutable snapshot. A completed old write never causes a newer pending
revision to be forgotten.

## Storage and integrity

Each application session has a random, private session identifier and two atomic generation
slots. A small recovery envelope records:

- a recovery-format version independent of the `.nanoscope` manifest version;
- the session identifier and monotonically increasing recovery revision;
- creation time and, when applicable, the source project path;
- the explicit-save revision covered when the snapshot was made; and
- the ordinary NanoScope project payload.

The payload uses the same serializer and `ProjectState` schema as an explicit project save.
Raw recordings remain references and are not copied into recovery storage. Generation files
are written atomically. Candidate discovery validates the envelope and payload; unsupported,
truncated, or corrupt files are skipped, and the older generation is tried next.

Recovery storage is private to the current user. Old sessions are bounded by retention
cleanup so abandoned checkpoints do not grow without limit. Multiple running NanoScope
windows/processes have distinct session identifiers and cannot retire one another's data.

## Revision-safe explicit saves

Before an explicit save gathers state, `MainWindow` records the revision represented by that
snapshot. After the guarded project write commits successfully, it retires recovery data only
through that exact revision. This is a fence, not a blanket delete:

```text
revision 12  explicit-save snapshot gathered
revision 13  user edits while the write is in flight
revision 12  project commit succeeds

retire <= 12; retain revision 13
```

The same rule covers the simpler cases:

| Save outcome | Recovery retirement |
| --- | --- |
| Successful save, no newer edit | Retire generations through the committed revision. |
| Successful save, newer edit exists | Retire only through the committed revision; keep the newer snapshot/session eligible. |
| Cancelled or failed save | Retire nothing. |
| Save intentionally omits live work | Keep the omitted/newer work recoverable and visibly dirty. |

## Shutdown rules

A normal shutdown stops the idle/max timers and drains or detaches the one bounded recovery
writer through NanoScope's existing worker-lifetime protocol. It does not create a false
"clean" state. Unsaved recovery data remains on disk unless an explicit save covered it or
the user explicitly discarded it. A crash requires no shutdown hook; the last atomically
completed generation is the recovery point.

## Validation matrix

The Stage 2 regression suite covers these externally meaningful cases:

- idle deadline and hard maximum deadline;
- immutable GUI-thread snapshot handed to a background serializer;
- one writer with latest-revision coalescing;
- valid newest generation and corrupt-newest fallback;
- Recover, Discard, Show in Finder, and cancel/close behaviour;
- recovered state is visibly dirty on the main window and child editing windows;
- successful, failed, cancelled, and overlapped explicit saves;
- revision-fenced retirement when an edit lands during save;
- isolation of two recoverable sessions; and
- timer/writer shutdown without deleting still-needed recovery data.

## Explicitly outside Stage 2

Stage 2 does not persist undo/redo history, copy raw recordings, sync recovery data to the
cloud, create portable project bundles, add immutable event UUIDs, or implement reviewer
audit history. Those remain later-stage work. A recovery snapshot is an emergency checkpoint;
the durable, shareable artifact is still a user-named `.nanoscope` project.
