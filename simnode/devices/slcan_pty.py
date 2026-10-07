"""slcan adapter on a pty: e_comms_node opens the slave with python-can, the sim reads and writes the master."""
import os
import tty

REPLIES = {b"O": b"\r", b"C": b"\r", b"S6": b"\r", b"V": b"V1013\r", b"F": b"F00\r", b"": b"\r"}


def encode_std(can_id, data):
    return b"t%03X%d" % (can_id, len(data)) + data.hex().upper().encode() + b"\r"


def encode_ext(can_id, data):
    return b"T%08X%d" % (can_id, len(data)) + data.hex().upper().encode() + b"\r"


def parse_line(line):
    """(id, is_ext, data) for a t/T frame line, else None."""
    s = line.rstrip(b"\r")
    try:
        if s[:1] == b"t":
            n = int(s[4:5])
            return int(s[1:4], 16), False, bytes.fromhex(s[5:5 + 2 * n].decode())
        if s[:1] == b"T":
            n = int(s[9:10])
            return int(s[1:9], 16), True, bytes.fromhex(s[10:10 + 2 * n].decode())
    except ValueError:
        return None
    return None


class SlcanPty:
    def __init__(self):
        self.master, self.slave = os.openpty()
        tty.setraw(self.slave)
        os.set_blocking(self.master, False)
        self.path = os.ttyname(self.slave)
        self.opened = False
        self._buf = b""

    def poll(self):
        """Frames the client sent since the last call; handshake commands are answered in place."""
        while True:
            try:
                chunk = os.read(self.master, 4096)
            except (BlockingIOError, OSError):
                break
            if not chunk:
                break
            self._buf += chunk
        frames = []
        *lines, self._buf = self._buf.split(b"\r")
        for line in lines:
            f = parse_line(line)
            if f is not None:
                frames.append(f)
                continue
            if line == b"O":
                self.opened = True
            elif line == b"C":
                self.opened = False
            reply = REPLIES.get(line, b"\r" if line[:1] in (b"S", b"s", b"L") else None)
            if reply is not None:
                self._write(reply)
        return frames

    def _write(self, b):
        try:
            os.write(self.master, b)
        except (BlockingIOError, OSError):
            pass  # a client that stopped reading drops frames, as a full CAN adapter buffer would

    def send_std(self, can_id, data):
        if self.opened:
            self._write(encode_std(can_id, data))

    def send_ext(self, can_id, data):
        if self.opened:
            self._write(encode_ext(can_id, data))

    def close(self):
        for fd in (self.master, self.slave):
            try:
                os.close(fd)
            except OSError:
                pass
