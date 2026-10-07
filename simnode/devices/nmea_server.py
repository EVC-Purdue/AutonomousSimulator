"""UM982 stand-in: NMEA lines served over TCP on 127.0.0.1, as the kart's socat bridge serves the receiver."""
import select
import socket


class NmeaServer:
    def __init__(self, port=0):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", port))
        self.sock.listen(4)
        self.sock.setblocking(False)
        self.port = self.sock.getsockname()[1]
        self.clients = []
        self.pending = {}  # client -> bytes not yet accepted by its socket, so no line is ever cut
        self.received = b""  # what clients sent (UNLOG / LOG commands), kept only for inspection

    def poll(self):
        """Accepts new clients and reads and discards what they send."""
        ready = select.select([self.sock] + self.clients, [], [], 0)[0]  # one syscall instead of two failing ones per poll
        while self.sock in ready:
            try:
                c, _ = self.sock.accept()
            except (BlockingIOError, OSError):
                break
            c.setblocking(False)
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.clients.append(c)
            self.pending[c] = b""
        for c in [c for c in self.clients if c in ready]:
            while True:
                try:
                    b = c.recv(4096)
                except BlockingIOError:
                    break
                except OSError:
                    b = b""
                if not b:
                    self._drop(c)
                    break
                self.received = (self.received + b)[-4096:]
        for c in list(self.clients):
            self._flush(c)

    def _drop(self, c):
        self.clients.remove(c)
        self.pending.pop(c, None)
        c.close()

    def _flush(self, c):
        buf = self.pending[c]
        while buf:
            try:
                n = c.send(buf)
            except BlockingIOError:
                break
            except OSError:
                self._drop(c)
                return
            buf = buf[n:]
        self.pending[c] = buf

    def write(self, lines):
        self.poll()
        if not lines:
            return
        data = "".join(lines).encode("ascii")
        for c in list(self.clients):
            self.pending[c] += data
            self._flush(c)

    def close(self):
        for c in list(self.clients):
            self._drop(c)
        self.sock.close()
