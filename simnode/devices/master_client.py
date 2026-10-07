"""Client for master_api's HTTP state endpoints (api_port, 8000 on the kart)."""
import json
import time
import urllib.request


class MasterClient:
    def __init__(self, port=8000, tries=20, wait_s=0.25):
        self.url, self.tries, self.wait_s = f"http://127.0.0.1:{port}", tries, wait_s

    def _call(self, path, body=None):
        err = None
        for _ in range(self.tries):
            try:
                data = None if body is None else json.dumps(body).encode()
                req = urllib.request.Request(self.url + path, data=data, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=2.0) as r:
                    return json.loads(r.read().decode())
            except OSError as e:
                err = e
                time.sleep(self.wait_s)
        raise ConnectionError(f"master API {path}: {err!r}")

    def set_state(self, name):
        out = self._call("/set_state", {"state": name})
        if "success" not in out:
            raise RuntimeError(f"set_state {name}: {out}")
        return out

    def get_state(self):
        return self._call("/get_state")["state"]
