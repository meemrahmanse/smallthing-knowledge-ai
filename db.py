from __future__ import annotations
import time
import pickle
from typing import Any, Optional

try:
    import redis
    REDIS_AVAILABLE = True
except Exception:
    REDIS_AVAILABLE = False


class CacheBackend:
    def get(self, key: str) -> Optional[Any]:
        raise NotImplementedError
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        raise NotImplementedError
    def delete(self, key: str) -> bool:
        raise NotImplementedError
    def clear(self, pattern: str = "") -> int:
        raise NotImplementedError
    def exists(self, key: str) -> bool:
        raise NotImplementedError


class MemoryCache(CacheBackend):
    def __init__(self, maxsize: int = 1000):
        self.cache = {}
        self.maxsize = maxsize

    def get(self, key: str):
        entry = self.cache.get(key)
        if not entry:
            return None
        value, expiry = entry
        if expiry and time.time() > expiry:
            del self.cache[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        if len(self.cache) >= self.maxsize:
            # simple eviction
            oldest = min(self.cache.items(), key=lambda i: i[1][1] or 0)[0]
            del self.cache[oldest]
        expiry = time.time() + ttl if ttl > 0 else None
        self.cache[key] = (value, expiry)
        return True

    def delete(self, key: str) -> bool:
        return self.cache.pop(key, None) is not None

    def clear(self, pattern: str = "") -> int:
        if not pattern:
            n = len(self.cache)
            self.cache.clear()
            return n
        todel = [k for k in self.cache.keys() if pattern in k]
        for k in todel:
            del self.cache[k]
        return len(todel)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None


class RedisCache(CacheBackend):
    def __init__(self, host='localhost', port=6379, db=0, prefix='smallthing:'):
        self.prefix = prefix
        self.client = redis.Redis(host=host, port=port, db=db)

    def _k(self, key: str) -> str:
        return f"{self.prefix}{key}"

    def get(self, key: str):
        data = self.client.get(self._k(key))
        if data is None:
            return None
        try:
            return pickle.loads(data)
        except Exception:
            return data.decode('utf-8')

    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        try:
            data = pickle.dumps(value)
            if ttl > 0:
                return self.client.setex(self._k(key), ttl, data)
            return self.client.set(self._k(key), data)
        except Exception:
            return False

    def delete(self, key: str) -> bool:
        return self.client.delete(self._k(key)) > 0

    def clear(self, pattern: str = "") -> int:
        keys = self.client.keys(self._k(pattern))
        if keys:
            return self.client.delete(*keys)
        return 0

    def exists(self, key: str) -> bool:
        return self.client.exists(self._k(key)) > 0


class CacheManager:
    def __init__(self, redis_url: Optional[str] = None):
        self.backend: CacheBackend
        if redis_url and REDIS_AVAILABLE:
            try:
                from urllib.parse import urlparse
                p = urlparse(redis_url)
                db = int(p.path.lstrip('/') or 0)
                self.backend = RedisCache(host=p.hostname or 'localhost', port=p.port or 6379, db=db)
                self.backend.client.ping()
            except Exception:
                self.backend = MemoryCache()
        else:
            self.backend = MemoryCache()

    def get(self, key: str):
        return self.backend.get(key)
    def set(self, key: str, value: Any, ttl: int = 300) -> bool:
        return self.backend.set(key, value, ttl)
    def delete(self, key: str) -> bool:
        return self.backend.delete(key)
    def clear(self, pattern: str = '') -> int:
        return self.backend.clear(pattern)


# singleton
import os
cachemanager = CacheManager(redis_url=os.getenv('REDISURL'))
