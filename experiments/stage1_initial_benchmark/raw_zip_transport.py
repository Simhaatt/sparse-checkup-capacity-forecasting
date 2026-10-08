"""Bounded public-source ZIP range transport, standalone for reproduction."""
import io,requests
URL="https://www.kaggle.com/api/v1/datasets/download/simhaa/luh-blank-result-v2"
class Ranged(io.RawIOBase):

    def __init__(self):
        self.session = requests.Session()
        r = self.session.get(URL, stream=True, timeout=40)
        r.raise_for_status()
        self.url = r.url
        self.size = int(r.headers['content-length'])
        self.pos = 0
        self.bytes = 0
        r.close()

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, n, whence=0):
        self.pos = n if whence == 0 else self.pos + n if whence == 1 else self.size + n
        return self.pos

    def read(self, n=-1):
        n = min(self.size - self.pos, n if n >= 0 else self.size - self.pos)
        if n <= 0:
            return b''
        if n > 64 * 1024 ** 2 or self.bytes + n > 150 * 1024 ** 2:
            raise ValueError('Bounded raw-data audit read limit exceeded')
        r = self.session.get(self.url, headers={'Range': f'bytes={self.pos}-{self.pos + n - 1}', 'Accept-Encoding': 'identity'}, stream=True, timeout=45)
        try:
            if r.status_code != 206:
                raise ValueError('Server does not support bounded byte ranges')
            data = r.raw.read(n + 1)
            if len(data) != n:
                raise ValueError('Incorrect range size')
            self.pos += n
            self.bytes += n
            return data
        finally:
            r.close()
