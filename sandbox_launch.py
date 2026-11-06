"""Linux-only, fail-closed bubblewrap launcher; no participant code runs here."""

import ctypes
import ctypes.util
import errno
import os
import resource
import site
import sys
import sysconfig
from pathlib import Path


def seccomp_filter():
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp is required; refusing unsandboxed execution.")
    lib = ctypes.CDLL(library)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    lib.seccomp_rule_add.restype = ctypes.c_int
    lib.seccomp_export_bpf.argtypes = [ctypes.c_void_p, ctypes.c_int]
    lib.seccomp_export_bpf.restype = ctypes.c_int
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(
        0x7FFF0000
    )  # SCMP_ACT_ALLOW; unsupported ABIs are killed.
    if not context:
        raise RuntimeError("Cannot construct seccomp filter.")

    class Comparison(ctypes.Structure):
        _fields_ = [
            ("arg", ctypes.c_uint),
            ("op", ctypes.c_int),
            ("a", ctypes.c_uint64),
            ("b", ctypes.c_uint64),
        ]

    def deny(name, error=errno.EPERM, comparisons=()):
        number = lib.seccomp_syscall_resolve_name(name.encode())
        if (
            number >= 0
            and lib.seccomp_rule_add(
                context, 0x00050000 | error, number, len(comparisons), *comparisons
            )
            < 0
        ):
            raise RuntimeError(f"Cannot restrict syscall {name}.")

    try:
        for name in (
            "fork",
            "vfork",
            "socket",
            "socketpair",
            "ptrace",
            "process_vm_readv",
            "process_vm_writev",
            "mount",
            "umount2",
            "pivot_root",
            "setns",
            "unshare",
            "open_by_handle_at",
            "name_to_handle_at",
            "move_mount",
            "fsopen",
            "fsconfig",
            "fsmount",
            "fspick",
            "mount_setattr",
            "bpf",
            "perf_event_open",
            "keyctl",
            "add_key",
            "request_key",
            "userfaultfd",
            "io_uring_setup",
            "io_uring_enter",
            "io_uring_register",
            "init_module",
            "finit_module",
            "delete_module",
            "kexec_load",
            "kexec_file_load",
            "reboot",
            "swapon",
            "swapoff",
            "sethostname",
            "setdomainname",
        ):
            deny(name)
        deny(
            "clone3", errno.ENOSYS
        )  # Let glibc fall back to constrained clone for threads.
        deny(
            "clone", comparisons=(Comparison(0, 7, 0x00010000, 0),)
        )  # No CLONE_THREAD => deny.
        descriptor = os.memfd_create("nqc-policy")
        if lib.seccomp_export_bpf(context, descriptor) < 0:
            raise RuntimeError("Cannot export seccomp filter.")
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.set_inheritable(descriptor, True)
        return descriptor
    finally:
        lib.seccomp_release(context)


def main():
    if sys.platform != "linux" or os.geteuid() == 0:
        raise RuntimeError("Run the worker as a dedicated, unprivileged Linux user.")
    bwrap, strategy, bootstrap = sys.argv[1:]
    policy = seccomp_filter()
    args = [
        bwrap,
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-pid",
        "--unshare-net",
        "--unshare-uts",
        "--disable-userns",
        "--new-session",
        "--die-with-parent",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--hostname",
        "nqc-strategy",
        "--dir",
        "/work",
        "--dir",
        "/tmp",
        "--chdir",
        "/work",
        "--dev",
        "/dev",
        "--proc",
        "/proc",
        "--remount-ro",
        "/dev",
        "--remount-ro",
        "/proc",
    ]
    # Bind runtime components, never /, /home, /etc, /app, or the project tree.
    # Production uses a dedicated strategy venv with no application packages.
    roots = [sysconfig.get_path("stdlib"), *site.getsitepackages()]
    roots += [
        "/usr/lib/x86_64-linux-gnu",
        "/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/lib/aarch64-linux-gnu",
        "/lib64",
        "/usr/lib64",
        "/usr/share/zoneinfo",
        "/usr/lib/locale",
    ]
    if sys.prefix != sys.base_prefix:
        roots.append(sys.prefix)
    for root in sorted(set(roots)):
        path = Path(root)
        if path.is_dir():
            args += ["--ro-bind", str(path), str(path)]
            # Debian numerical libraries can use /etc/alternatives symlinks.
            # Expose the individual library targets, never the /etc tree.
            for pattern in ("libblas.so.*", "liblapack.so.*"):
                for library in path.glob(pattern):
                    if library.is_symlink() and str(library.readlink()).startswith(
                        "/etc/alternatives/"
                    ):
                        args += [
                            "--ro-bind",
                            str(library.resolve()),
                            str(library.readlink()),
                        ]
    executable = str(Path(sys.executable).resolve())
    aliases = {executable, str(getattr(sys, "_base_executable", executable))}
    if sys.prefix == sys.base_prefix:
        aliases.add(sys.executable)
    for candidate in Path(executable).parent.glob("python*"):
        if candidate.is_file() and str(candidate.resolve()) == executable:
            aliases.add(str(candidate))
    for alias in sorted(aliases):
        args += ["--ro-bind", executable, alias]
    # Some Python builds link libpython outside the system's multiarch lib dir.
    libdir = Path(sysconfig.get_config_var("LIBDIR") or "/nonexistent")
    for library in sorted(libdir.glob("libpython*.so*")):
        args += ["--ro-bind", str(library), str(library)]
    args += [
        "--ro-bind",
        strategy,
        "/strategy.py",
        "--ro-bind",
        bootstrap,
        "/runner.py",
    ]
    for key, value in {
        "PATH": "/usr/bin",
        "HOME": "/work",
        "LANG": "C.UTF-8",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_MAX_THREADS": "1",
        "NUMBA_NUM_THREADS": "1",
    }.items():
        args += ["--setenv", key, value]
    args += [
        "--remount-ro",
        "/",
        "--seccomp",
        str(policy),
        sys.executable,
        "-I",
        "-B",
        "/runner.py",
        "/strategy.py",
    ]
    for limit, value in (
        (resource.RLIMIT_AS, 1024**3),
        (resource.RLIMIT_CPU, 30),
        (resource.RLIMIT_NPROC, 64),
        (resource.RLIMIT_NOFILE, 64),
        (resource.RLIMIT_FSIZE, 1024**2),
        (resource.RLIMIT_CORE, 0),
    ):
        resource.setrlimit(limit, (value, value))
    os.execve(bwrap, args, {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})


if __name__ == "__main__":
    main()
