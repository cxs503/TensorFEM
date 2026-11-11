import re

p = 'src/tensorfem/parallel/comm.py'
s = open(p).read()
s = s.replace("""        self._c = dict(conns)
        self._have_peers = True""", """        self._c = dict(conns)
        self._pool = None""")
s = s.replace("""        from concurrent.futures import ThreadPoolExecutor
        out = {}
        with ThreadPoolExecutor(max_workers=len(peers)) as ex:
            futs = {j: ex.submit(self.sendrecv, sends.get(j), j,
                                 int(recv_sizes.get(j, 0)) * 8, j)
                    for j in peers}
            for j, f in futs.items():
                out[j] = f.result()
        return out""", """        if self._pool is None:
            from concurrent.futures import ThreadPoolExecutor
            self._pool = ThreadPoolExecutor(max_workers=max(1, self.size - 1))
        pool = self._pool
        futs = {j: pool.submit(self.sendrecv, sends.get(j), j,
                               int(recv_sizes.get(j, 0)) * 8, j) for j in peers}
        return {j: f.result() for j, f in futs.items()}""")
s = s.replace("""    def close(self) -> None:
        for c in self._c.values():""", """    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        for c in self._c.values():""")
open(p, 'w').write(s)
print("ok")