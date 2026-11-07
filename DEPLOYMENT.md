# Coolify/Nixpacks and a native strategy worker

The web app remains a normal Python application deployed by Coolify/Nixpacks.
The separate strategy worker is a native Linux service. Do not add privileges,
host filesystem mounts, a Docker socket, or relaxed namespace/seccomp settings
to the web container in order to run uploaded code there.

These are provisioning instructions, not a record of changes made to the live
server. The live Hetzner/Coolify environment must be checked by its operator
before enabling the runner.

## Native worker

Use a patched Linux host with unprivileged user namespaces, bubblewrap 0.9+,
libseccomp, Python 3.10+, and a dedicated non-root service account. A separate
host is also supported. The service may run alongside Coolify on the existing
host, but it is independent of the web application's execution environment.

On Ubuntu/Debian, the operator can install the prerequisites and create a service
account:

```sh
sudo apt-get install bubblewrap libseccomp2 python3-venv
sudo useradd --system --create-home --home-dir /opt/nqc-worker --shell /usr/sbin/nologin nqc-worker
```

Place `strategy_runner.py`, `sandbox_launch.py`, `strategy_bootstrap.py`, and
`requirements-worker.txt` in `/opt/nqc-worker`. Create a dedicated venv there
and install the worker requirements. Keep application source, database files,
round data, credentials, and other private files out of this venv and the system
Python library directories. The guest receives the venv and selected system
runtime libraries, never the worker's entire working directory.

```sh
sudo python3 -m venv /opt/nqc-worker/.venv
sudo /opt/nqc-worker/.venv/bin/pip install -r /opt/nqc-worker/requirements-worker.txt
sudo /opt/nqc-worker/.venv/bin/pip freeze > worker-runtime-lock.txt
```

Keep the resolved packages fixed for each round. The runtime fingerprint
includes Python, installed strategy libraries, and the runner/launcher/bootstrap
code. Upgrading any of them requires checking compatibility and starting a new
round or restoring the previous runtime; it must not silently change old scores.
Restart the service whenever code or packages change.

The client and worker negotiate incremental history updates. The guest receives
the initial observation prefix and subsequent new rows, and reconstructs the
complete available history for each strategy call. Update the worker files and
restart the service when deploying this transport revision. Profile the full
public practice dataset through the deployed HTTPS route against the session limits.

Generate a random worker credential and place `NQC_WORKER_TOKEN=<value>` in
`/etc/nqc-worker.env`, with root ownership and mode `0600`. Keep the token out of
Git, command-line arguments, service URLs, and logs. Give the web app the same
token through Coolify's environment configuration.

Review and install the supplied `deploy/nqc-worker.service` unit. It binds to
localhost by default, drops capabilities, limits service memory/CPU/tasks, and
uses a private temporary directory. Run the tests as the unprivileged worker
user before starting it. If isolation is blocked by the host's kernel or
security policy, resolve that with the server operator; there is no permissive
execution fallback.

## Connect Coolify to the worker

The API uses bearer authentication and refuses unauthenticated requests. The
web client requires HTTPS except for actual loopback addresses.

The operator must configure a TLS reverse proxy that reaches the native service.
The systemd unit intentionally starts with a localhost listener. Depending on
the existing proxy arrangement, use a host-side TLS proxy, or bind the worker
only to the appropriate private interface reachable by Coolify's proxy and
restrict that port to the proxy. A container's `127.0.0.1` is not the host's
loopback, so do not enter a localhost URL in Coolify unless the worker really
runs in that same network namespace. Do not expose the plaintext worker port
directly to the internet.

Set the web application environment:

```text
ADMIN_PASSWORD=<independent random password of at least 32 characters>
NQC_WORKER_URL=https://<your worker's TLS endpoint>
NQC_WORKER_TOKEN=<worker credential>
DB_PATH=/app/data/leaderboard.db
ROUND_ROOT=/app/data/rounds
```

Persist `/app/data` using the existing Coolify storage configuration. The web app
does not need namespace privileges or any access to the worker's filesystem.
The worker must not receive `/app/data`, the final dataset, or web app secrets.

## Prepare and operate a round

1. Put the chosen public training CSVs in `data/training` or another organizer
   folder. Data preparation is separate from strategy execution and scoring.
2. Enter the organizer password, create teams, and distribute their tokens.
3. Create a round from the public folder. The app copies a public snapshot,
   generates a separate private final snapshot with a fresh 128-bit seed, pins
   the runtime/configuration, and opens submissions.
4. Teams practice on the public snapshot, store versions, and select a final one.
5. Lock submissions, evaluate, inspect failures, retry failed jobs if needed,
   and publish only when the round is ready. An active database lease prevents
   two admins from scoring the same round simultaneously. A crashed lease
   expires after the maximum remaining time budget for one team.
6. Keep the private bundle and its seed private. It may be revealed after the
   competition if independent verification is desired; never reuse a revealed
   seed or final dataset for a new open round.

## Limits and threat model

Each sandbox has a 1 GiB address-space limit, 30 CPU seconds, a 45-second total
lifetime, and a three-second response limit per bar. Startup has 15 seconds.
The service admits two sandbox sessions and 16 HTTP request threads. Individual
responses are capped at 256 KiB; persistent trader data at 100 KB; source at
128 KB. Both normal and native-library code are subject to the OS policy.

The guest starts with an empty root, read-only runtime mounts, private process,
user, IPC, UTS, and network namespaces, no inherited environment, and a seccomp
policy blocking sockets, process creation, nested namespaces, ptrace, and other
privileged interfaces. Constrained threads are allowed for numerical libraries.
The filesystem, including its temporary directory, is read-only. Strategies
must avoid file caches and network calls; memory-only computations work.

The worker is an OS sandbox sharing a kernel, not a virtual machine. Keep the
host, bubblewrap, Python, and dependencies patched. The operator must verify
the real server, TLS routing, package compatibility, cgroup resource limits,
and the chosen public dataset before enabling production evaluations. Local
tests cannot establish those properties for a server we have not accessed.
