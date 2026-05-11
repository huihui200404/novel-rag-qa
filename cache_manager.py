from cachetools import TTLCache
import hashlib

class CacheManager:
    def __init__(self, maxsize=100, ttl=3600):
        self.cache = TTLCache(maxsize=maxsize, ttl=ttl)
    def _key(self, text): return hashlib.md5(text.encode()).hexdigest()
    def get(self, q): return self.cache.get(self._key(q))
    def set(self, q, val): self.cache[self._key(q)] = val