"""Opt-in, disjoint TCP port ranges for single-node SGLang replicas."""

import errno
import socket


def reserve_replica_port(base: int, span: int, replica_rank: int):
    """Return (port, listening socket); caller closes just before engine launch.

    Ranges are disjoint within a job. This cannot atomically hand the socket to
    PyTorch TCPStore, so unrelated processes can still race after its release.
    Concurrent jobs sharing a network namespace should use different bases.
    """
    if base < 1024 or span < 1 or replica_rank < 0:
        raise ValueError("NCCL port base must be >=1024, span >0, and replica rank >=0")
    start = base + span * replica_rank
    end = start + span
    if end > 65536:
        raise ValueError(f"Replica {replica_rank} port range exceeds 65535: [{start}, {end})")
    for port in range(start, end):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(("0.0.0.0", port))
            sock.listen(1)
        except OSError as exc:
            sock.close()
            if exc.errno == errno.EADDRINUSE:
                continue
            raise
        return port, sock
    raise RuntimeError(f"No free SGLang NCCL port for replica {replica_rank} in [{start}, {end})")
