#!/usr/bin/env python3
"""
VectorDB Engine - Python port (standard library only)

Features
  * Brute force, KD-Tree and HNSW nearest-neighbour search
  * 16D demo dataset + benchmark + HNSW graph inspection
  * Document store: chunking + Ollama embeddings + HNSW (cosine)
  * RAG endpoint: embed -> retrieve -> generate

Ollama setup:
  https://ollama.com
  ollama pull nomic-embed-text
  ollama pull llama3.2

Run:  python vectordb.py     (serves http://localhost:8080, expects index.html next to it)
"""

import heapq
import json
import math
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

DIMS = 16  # demo vectors; doc embedding dimension comes from Ollama at runtime


# =====================================================================
#  DATA TYPES
# =====================================================================

class VectorItem:
    __slots__ = ("id", "metadata", "category", "emb")

    def __init__(self, id, metadata, category, emb):
        self.id = id
        self.metadata = metadata
        self.category = category
        self.emb = emb


# =====================================================================
#  DISTANCE METRICS
# =====================================================================

def euclidean(a, b):
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na < 1e-9 or nb < 1e-9:
        return 1.0
    return 1.0 - dot / (math.sqrt(na) * math.sqrt(nb))


def manhattan(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def get_dist_fn(name):
    if name == "cosine":
        return cosine
    if name == "manhattan":
        return manhattan
    return euclidean


# =====================================================================
#  BRUTE FORCE
# =====================================================================

class BruteForce:
    def __init__(self):
        self.items = []

    def insert(self, v):
        self.items.append(v)

    def knn(self, q, k, dist):
        r = sorted((dist(q, v.emb), v.id) for v in self.items)
        return r[:k]

    def remove(self, id):
        self.items = [v for v in self.items if v.id != id]


# =====================================================================
#  KD-TREE
# =====================================================================

class KDNode:
    __slots__ = ("item", "left", "right")

    def __init__(self, item):
        self.item = item
        self.left = None
        self.right = None


class KDTree:
    def __init__(self, dims):
        self.dims = dims
        self.root = None

    def insert(self, v):
        node = KDNode(v)
        if self.root is None:
            self.root = node
            return
        cur, d = self.root, 0
        while True:
            ax = d % self.dims
            if v.emb[ax] < cur.item.emb[ax]:
                if cur.left is None:
                    cur.left = node
                    return
                cur = cur.left
            else:
                if cur.right is None:
                    cur.right = node
                    return
                cur = cur.right
            d += 1

    def _knn(self, n, q, k, d, dist, heap):
        # heap holds (-distance, id) so heap[0] is the current worst hit
        if n is None:
            return
        dn = dist(q, n.item.emb)
        if len(heap) < k or dn < -heap[0][0]:
            heapq.heappush(heap, (-dn, n.item.id))
            if len(heap) > k:
                heapq.heappop(heap)
        ax = d % self.dims
        diff = q[ax] - n.item.emb[ax]
        closer, farther = (n.left, n.right) if diff < 0 else (n.right, n.left)
        self._knn(closer, q, k, d + 1, dist, heap)
        if len(heap) < k or abs(diff) < -heap[0][0]:
            self._knn(farther, q, k, d + 1, dist, heap)

    def knn(self, q, k, dist):
        heap = []
        self._knn(self.root, q, k, 0, dist, heap)
        return sorted((-nd, id) for nd, id in heap)

    def rebuild(self, items):
        self.root = None
        for v in items:
            self.insert(v)


# =====================================================================
#  HNSW - Hierarchical Navigable Small World
# =====================================================================

class HNSWNode:
    __slots__ = ("item", "max_lyr", "nbrs")

    def __init__(self, item, max_lyr):
        self.item = item
        self.max_lyr = max_lyr
        self.nbrs = [[] for _ in range(max_lyr + 1)]


class HNSW:
    def __init__(self, m=16, ef_build=200):
        self.M = m
        self.M0 = 2 * m
        self.ef_build = ef_build
        self.mL = 1.0 / math.log(m)
        self.top_layer = -1
        self.entry_pt = -1
        self.G = {}
        self.rng = random.Random(42)

    def _rand_level(self):
        u = 1.0 - self.rng.random()  # (0, 1]
        return int(math.floor(-math.log(u) * self.mL))

    def _search_layer(self, q, ep, ef, lyr, dist):
        G = self.G
        d0 = dist(q, G[ep].item.emb)
        visited = {ep}
        cands = [(d0, ep)]          # min-heap
        found = [(-d0, ep)]         # max-heap via negation

        while cands:
            cd, cid = heapq.heappop(cands)
            if len(found) >= ef and cd > -found[0][0]:
                break
            node = G[cid]
            if lyr >= len(node.nbrs):
                continue
            for nid in node.nbrs[lyr]:
                if nid in visited or nid not in G:
                    continue
                visited.add(nid)
                nd = dist(q, G[nid].item.emb)
                if len(found) < ef or nd < -found[0][0]:
                    heapq.heappush(cands, (nd, nid))
                    heapq.heappush(found, (-nd, nid))
                    if len(found) > ef:
                        heapq.heappop(found)

        return sorted((-nd, id) for nd, id in found)

    @staticmethod
    def _select_nbrs(cands, max_m):
        return [cid for _, cid in cands[:max_m]]

    def insert(self, item, dist):
        G = self.G
        id_ = item.id
        lvl = self._rand_level()
        G[id_] = HNSWNode(item, lvl)

        if self.entry_pt == -1:
            self.entry_pt = id_
            self.top_layer = lvl
            return

        ep = self.entry_pt
        for lc in range(self.top_layer, lvl, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(item.emb, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]

        for lc in range(min(self.top_layer, lvl), -1, -1):
            W = self._search_layer(item.emb, ep, self.ef_build, lc, dist)
            max_m = self.M0 if lc == 0 else self.M
            sel = self._select_nbrs(W, max_m)
            G[id_].nbrs[lc] = sel

            for nid in sel:
                if nid not in G:
                    continue
                nbrs = G[nid].nbrs
                while len(nbrs) <= lc:
                    nbrs.append([])
                conn = nbrs[lc]
                conn.append(id_)
                if len(conn) > max_m:
                    ds = sorted(
                        (dist(G[nid].item.emb, G[c].item.emb), c)
                        for c in conn if c in G
                    )
                    nbrs[lc] = [c for _, c in ds[:max_m]]
            if W:
                ep = W[0][1]

        if lvl > self.top_layer:
            self.top_layer = lvl
            self.entry_pt = id_

    def knn(self, q, k, ef, dist):
        if self.entry_pt == -1:
            return []
        G = self.G
        ep = self.entry_pt
        for lc in range(self.top_layer, 0, -1):
            if lc < len(G[ep].nbrs):
                W = self._search_layer(q, ep, 1, lc, dist)
                if W:
                    ep = W[0][1]
        W = self._search_layer(q, ep, max(ef, k), 0, dist)
        return W[:k]

    def remove(self, id_):
        G = self.G
        if id_ not in G:
            return
        for node in G.values():
            for i, layer in enumerate(node.nbrs):
                node.nbrs[i] = [x for x in layer if x != id_]
        if self.entry_pt == id_:
            self.entry_pt = -1
            for nid in G:
                if nid != id_:
                    self.entry_pt = nid
                    break
        del G[id_]
        # keep top_layer consistent with the new entry point
        if self.entry_pt == -1:
            self.top_layer = -1
        else:
            self.top_layer = max(n.max_lyr for n in G.values())
            for nid, n in G.items():
                if n.max_lyr == self.top_layer:
                    self.entry_pt = nid
                    break

    def get_info(self):
        max_l = max(self.top_layer + 1, 1)
        nodes_per_layer = [0] * max_l
        edges_per_layer = [0] * max_l
        nodes, edges = [], []
        for id_, nd in self.G.items():
            nodes.append({
                "id": id_,
                "metadata": nd.item.metadata,
                "category": nd.item.category,
                "maxLyr": nd.max_lyr,
            })
            for lc in range(min(nd.max_lyr, max_l - 1) + 1):
                nodes_per_layer[lc] += 1
                if lc < len(nd.nbrs):
                    for nid in nd.nbrs[lc]:
                        if id_ < nid:
                            edges_per_layer[lc] += 1
                            edges.append({"src": id_, "dst": nid, "lyr": lc})
        return {
            "topLayer": self.top_layer,
            "nodeCount": len(self.G),
            "nodesPerLayer": nodes_per_layer,
            "edgesPerLayer": edges_per_layer,
            "nodes": nodes,
            "edges": edges,
        }

    def size(self):
        return len(self.G)


# =====================================================================
#  VECTOR DATABASE  (demo 16D index)
# =====================================================================

class VectorDB:
    def __init__(self, dims):
        self.dims = dims
        self.store = {}
        self.bf = BruteForce()
        self.kdt = KDTree(dims)
        self.hnsw = HNSW(16, 200)
        self.mu = threading.RLock()
        self.next_id = 1

    def insert(self, meta, cat, emb, dist):
        with self.mu:
            v = VectorItem(self.next_id, meta, cat, emb)
            self.next_id += 1
            self.store[v.id] = v
            self.bf.insert(v)
            self.kdt.insert(v)
            self.hnsw.insert(v, dist)
            return v.id

    def remove(self, id_):
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.bf.remove(id_)
            self.hnsw.remove(id_)
            self.kdt.rebuild(list(self.store.values()))
            return True

    def search(self, q, k, metric, algo):
        with self.mu:
            dfn = get_dist_fn(metric)
            t0 = time.perf_counter()
            if algo == "bruteforce":
                raw = self.bf.knn(q, k, dfn)
            elif algo == "kdtree":
                raw = self.kdt.knn(q, k, dfn)
            else:
                raw = self.hnsw.knn(q, k, 50, dfn)
            us = int((time.perf_counter() - t0) * 1e6)

            hits = []
            for d, id_ in raw:
                v = self.store.get(id_)
                if v:
                    hits.append({"id": id_, "meta": v.metadata, "cat": v.category,
                                 "emb": v.emb, "dist": d})
            return {"hits": hits, "us": us, "algo": algo, "metric": metric}

    def benchmark(self, q, k, metric):
        with self.mu:
            dfn = get_dist_fn(metric)

            def timed(fn):
                t = time.perf_counter()
                fn()
                return int((time.perf_counter() - t) * 1e6)

            return {
                "bfUs": timed(lambda: self.bf.knn(q, k, dfn)),
                "kdUs": timed(lambda: self.kdt.knn(q, k, dfn)),
                "hnswUs": timed(lambda: self.hnsw.knn(q, k, 50, dfn)),
                "n": len(self.store),
            }

    def all(self):
        with self.mu:
            return list(self.store.values())

    def hnsw_info(self):
        with self.mu:
            return self.hnsw.get_info()

    def size(self):
        with self.mu:
            return len(self.store)


# =====================================================================
#  TEXT CHUNKER
# =====================================================================

def chunk_text(text, chunk_words=250, overlap_words=30):
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [text]

    chunks = []
    step = chunk_words - overlap_words
    for i in range(0, len(words), step):
        end = min(i + chunk_words, len(words))
        chunks.append(" ".join(words[i:end]))
        if end == len(words):
            break
    return chunks


# =====================================================================
#  OLLAMA CLIENT - wraps local Ollama REST API
# =====================================================================

class OllamaClient:
    def __init__(self, host="127.0.0.1", port=11434):
        self.base = f"http://{host}:{port}"
        self.embed_model = "nomic-embed-text"
        self.gen_model = "llama3.2"

    def _post(self, path, payload, timeout):
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8"))

    def is_available(self):
        try:
            with urllib.request.urlopen(self.base + "/api/tags", timeout=2) as r:
                return r.status == 200
        except Exception:
            return False

    def embed(self, text):
        """Returns [] if Ollama is not running or the model is missing."""
        try:
            data = self._post("/api/embeddings",
                              {"model": self.embed_model, "prompt": text}, 30)
            return [float(x) for x in data.get("embedding", [])] if data else []
        except Exception:
            return []

    def generate(self, prompt):
        try:
            data = self._post("/api/generate",
                              {"model": self.gen_model, "prompt": prompt,
                               "stream": False}, 180)
            if not data:
                raise RuntimeError
            return data.get("response", "")
        except Exception:
            return "ERROR: Ollama unavailable. Run: ollama serve"


# =====================================================================
#  DOCUMENT DATABASE - HNSW over real Ollama embeddings
# =====================================================================

class DocItem:
    __slots__ = ("id", "title", "text", "emb")

    def __init__(self, id, title, text, emb):
        self.id = id
        self.title = title
        self.text = text
        self.emb = emb


class DocumentDB:
    def __init__(self):
        self.store = {}
        self.hnsw = HNSW(16, 200)
        self.bf = BruteForce()
        self.mu = threading.RLock()
        self.next_id = 1
        self.dims = 0

    def insert(self, title, text, emb):
        with self.mu:
            if self.dims == 0:
                self.dims = len(emb)
            item = DocItem(self.next_id, title, text, emb)
            self.next_id += 1
            self.store[item.id] = item
            vi = VectorItem(item.id, title, "doc", emb)
            self.hnsw.insert(vi, cosine)
            self.bf.insert(vi)
            return item.id

    def search(self, q, k, max_dist=0.7):
        with self.mu:
            if not self.store:
                return []
            if len(self.store) < 10:
                raw = self.bf.knn(q, k, cosine)
            else:
                raw = self.hnsw.knn(q, k, 50, cosine)
            return [(d, self.store[id_]) for d, id_ in raw
                    if id_ in self.store and d <= max_dist]

    def remove(self, id_):
        with self.mu:
            if id_ not in self.store:
                return False
            del self.store[id_]
            self.hnsw.remove(id_)
            self.bf.remove(id_)
            return True

    def all(self):
        with self.mu:
            return list(self.store.values())

    def size(self):
        with self.mu:
            return len(self.store)


# =====================================================================
#  DEMO DATA  (16D categorical vectors)
# =====================================================================

def load_demo(db):
    dist = get_dist_fn("cosine")
    # Dims 0-3: CS | Dims 4-7: Math | Dims 8-11: Food | Dims 12-15: Sports
    data = [
        ("Linked List: nodes connected by pointers", "cs",
         [0.90, 0.85, 0.72, 0.68, 0.12, 0.08, 0.15, 0.10, 0.05, 0.08, 0.06, 0.09, 0.07, 0.11, 0.08, 0.06]),
        ("Binary Search Tree: O(log n) search and insert", "cs",
         [0.88, 0.82, 0.78, 0.74, 0.15, 0.10, 0.08, 0.12, 0.06, 0.07, 0.08, 0.05, 0.09, 0.06, 0.07, 0.10]),
        ("Dynamic Programming: memoization overlapping subproblems", "cs",
         [0.82, 0.76, 0.88, 0.80, 0.20, 0.18, 0.12, 0.09, 0.07, 0.06, 0.08, 0.07, 0.08, 0.09, 0.06, 0.07]),
        ("Graph BFS and DFS: breadth and depth first traversal", "cs",
         [0.85, 0.80, 0.75, 0.82, 0.18, 0.14, 0.10, 0.08, 0.06, 0.09, 0.07, 0.06, 0.10, 0.08, 0.09, 0.07]),
        ("Hash Table: O(1) lookup with collision chaining", "cs",
         [0.87, 0.78, 0.70, 0.76, 0.13, 0.11, 0.09, 0.14, 0.08, 0.07, 0.06, 0.08, 0.07, 0.10, 0.08, 0.09]),
        ("Calculus: derivatives integrals and limits", "math",
         [0.12, 0.15, 0.18, 0.10, 0.91, 0.86, 0.78, 0.72, 0.08, 0.06, 0.07, 0.09, 0.07, 0.08, 0.06, 0.10]),
        ("Linear Algebra: matrices eigenvalues eigenvectors", "math",
         [0.20, 0.18, 0.15, 0.12, 0.88, 0.90, 0.82, 0.76, 0.09, 0.07, 0.08, 0.06, 0.10, 0.07, 0.08, 0.09]),
        ("Probability: distributions random variables Bayes theorem", "math",
         [0.15, 0.12, 0.20, 0.18, 0.84, 0.80, 0.88, 0.82, 0.07, 0.08, 0.06, 0.10, 0.09, 0.06, 0.09, 0.08]),
        ("Number Theory: primes modular arithmetic RSA cryptography", "math",
         [0.22, 0.16, 0.14, 0.20, 0.80, 0.85, 0.76, 0.90, 0.08, 0.09, 0.07, 0.06, 0.08, 0.10, 0.07, 0.06]),
        ("Combinatorics: permutations combinations generating functions", "math",
         [0.18, 0.20, 0.16, 0.14, 0.86, 0.78, 0.84, 0.80, 0.06, 0.07, 0.09, 0.08, 0.06, 0.09, 0.10, 0.07]),
        ("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.90, 0.86, 0.78, 0.72, 0.08, 0.06, 0.09, 0.07]),
        ("Sushi: vinegared rice raw fish and nori rolls", "food",
         [0.06, 0.08, 0.07, 0.09, 0.09, 0.06, 0.08, 0.07, 0.86, 0.90, 0.82, 0.76, 0.07, 0.09, 0.06, 0.08]),
        ("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food",
         [0.09, 0.07, 0.06, 0.08, 0.08, 0.09, 0.07, 0.06, 0.82, 0.78, 0.90, 0.84, 0.09, 0.07, 0.08, 0.06]),
        ("Tacos: corn tortillas with carnitas salsa and cilantro", "food",
         [0.07, 0.09, 0.08, 0.06, 0.06, 0.07, 0.09, 0.08, 0.78, 0.82, 0.86, 0.90, 0.06, 0.08, 0.07, 0.09]),
        ("Croissant: laminated pastry with buttery flaky layers", "food",
         [0.06, 0.07, 0.10, 0.09, 0.10, 0.06, 0.07, 0.10, 0.85, 0.80, 0.76, 0.82, 0.09, 0.07, 0.10, 0.06]),
        ("Basketball: fast-paced shooting dribbling slam dunks", "sports",
         [0.09, 0.07, 0.08, 0.10, 0.08, 0.09, 0.07, 0.06, 0.08, 0.07, 0.09, 0.06, 0.91, 0.85, 0.78, 0.72]),
        ("Football: tackles touchdowns field goals and strategy", "sports",
         [0.07, 0.09, 0.06, 0.08, 0.09, 0.07, 0.10, 0.08, 0.07, 0.09, 0.08, 0.07, 0.87, 0.89, 0.82, 0.76]),
        ("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports",
         [0.08, 0.06, 0.09, 0.07, 0.07, 0.08, 0.06, 0.09, 0.09, 0.06, 0.07, 0.08, 0.83, 0.80, 0.88, 0.82]),
        ("Chess: openings endgames tactics strategic board game", "sports",
         [0.25, 0.20, 0.22, 0.18, 0.22, 0.18, 0.20, 0.15, 0.06, 0.08, 0.07, 0.09, 0.80, 0.84, 0.78, 0.90]),
        ("Swimming: butterfly freestyle backstroke Olympic competition", "sports",
         [0.06, 0.08, 0.07, 0.09, 0.08, 0.06, 0.09, 0.07, 0.10, 0.08, 0.06, 0.07, 0.85, 0.82, 0.86, 0.80]),
    ]
    for meta, cat, emb in data:
        db.insert(meta, cat, emb, dist)


# =====================================================================
#  HELPERS
# =====================================================================

def parse_vec(s):
    out = []
    for t in s.split(","):
        try:
            out.append(float(t))
        except ValueError:
            pass
    return out


def parse_int(s, default):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def r4(v):
    return [round(x, 4) for x in v]


# =====================================================================
#  HTTP SERVER
# =====================================================================

db = VectorDB(DIMS)
doc_db = DocumentDB()
ollama = OllamaClient()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ---- plumbing -----------------------------------------------------
    def log_message(self, fmt, *args):  # quiet
        pass

    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _send(self, body, ctype="application/json", status=200):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status=200):
        self._send(json.dumps(obj, ensure_ascii=False), status=status)

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        try:
            data = json.loads(raw.decode("utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    # ---- verbs --------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        u = urlparse(self.path)
        p = u.path
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}

        if p == "/":
            return self._index()
        if p == "/search":
            return self._search(qs)
        if p == "/items":
            return self._items()
        if p == "/benchmark":
            return self._benchmark(qs)
        if p == "/hnsw-info":
            return self._json(db.hnsw_info())
        if p == "/doc/list":
            return self._doc_list()
        if p == "/status":
            return self._status()
        if p == "/stats":
            return self._json({
                "count": db.size(), "dims": DIMS,
                "algorithms": ["bruteforce", "kdtree", "hnsw"],
                "metrics": ["euclidean", "cosine", "manhattan"],
            })
        self._send("not found", "text/plain", 404)

    def do_POST(self):
        p = urlparse(self.path).path
        body = self._body()
        if p == "/insert":
            return self._insert(body)
        if p == "/doc/insert":
            return self._doc_insert(body)
        if p == "/doc/search":
            return self._doc_search(body)
        if p == "/doc/ask":
            return self._doc_ask(body)
        self._send("not found", "text/plain", 404)

    def do_DELETE(self):
        p = urlparse(self.path).path
        m = re.fullmatch(r"/delete/(\d+)", p)
        if m:
            return self._json({"ok": db.remove(int(m.group(1)))})
        m = re.fullmatch(r"/doc/delete/(\d+)", p)
        if m:
            return self._json({"ok": doc_db.remove(int(m.group(1)))})
        self._send("not found", "text/plain", 404)

    # ---- demo vector endpoints ---------------------------------------
    def _index(self):
        try:
            with open("index.html", "rb") as f:
                self._send(f.read(), "text/html")
        except OSError:
            self._send("index.html not found", "text/plain", 404)

    def _search(self, qs):
        q = parse_vec(qs.get("v", ""))
        if len(q) != DIMS:
            return self._json({"error": f"need {DIMS}D vector"})
        k = parse_int(qs.get("k"), 5)
        metric = qs.get("metric") or "cosine"
        algo = qs.get("algo") or "hnsw"
        out = db.search(q, k, metric, algo)
        self._json({
            "results": [{
                "id": h["id"], "metadata": h["meta"], "category": h["cat"],
                "distance": round(h["dist"], 6), "embedding": r4(h["emb"]),
            } for h in out["hits"]],
            "latencyUs": out["us"], "algo": out["algo"], "metric": out["metric"],
        })

    def _insert(self, body):
        meta = body.get("metadata", "")
        cat = body.get("category", "")
        emb = body.get("embedding", [])
        try:
            emb = [float(x) for x in emb]
        except (TypeError, ValueError):
            emb = []
        if not meta or not emb or len(emb) != DIMS:
            return self._json({"error": "invalid body"})
        self._json({"id": db.insert(meta, cat, emb, get_dist_fn("cosine"))})

    def _items(self):
        self._json([{
            "id": v.id, "metadata": v.metadata,
            "category": v.category, "embedding": r4(v.emb),
        } for v in db.all()])

    def _benchmark(self, qs):
        q = parse_vec(qs.get("v", ""))
        if len(q) != DIMS:
            return self._json({"error": f"need {DIMS}D vector"})
        k = parse_int(qs.get("k"), 5)
        metric = qs.get("metric") or "cosine"
        b = db.benchmark(q, k, metric)
        self._json({"bruteforceUs": b["bfUs"], "kdtreeUs": b["kdUs"],
                    "hnswUs": b["hnswUs"], "itemCount": b["n"]})

    # ---- document + RAG endpoints ------------------------------------
    def _doc_insert(self, body):
        title = body.get("title", "")
        text = body.get("text", "")
        if not title or not text:
            return self._json({"error": "need title and text"})

        chunks = chunk_text(text, 250, 30)
        ids = []
        for i, chunk in enumerate(chunks):
            emb = ollama.embed(chunk)
            if not emb:
                return self._json({
                    "error": "Ollama unavailable. Install from https://ollama.com "
                             "then run: ollama pull nomic-embed-text && ollama pull llama3.2"
                })
            chunk_title = (f"{title} [{i + 1}/{len(chunks)}]"
                           if len(chunks) > 1 else title)
            ids.append(doc_db.insert(chunk_title, chunk, emb))

        self._json({"ids": ids, "chunks": len(chunks), "dims": doc_db.dims})

    def _doc_list(self):
        out = []
        for d in doc_db.all():
            preview = d.text[:120] + ("…" if len(d.text) > 120 else "")
            out.append({"id": d.id, "title": d.title, "preview": preview,
                        "words": len(d.text.split())})
        self._json(out)

    def _doc_search(self, body):
        question = body.get("question", "")
        k = parse_int(body.get("k"), 3)
        if not question:
            return self._json({"error": "need question"})
        q_emb = ollama.embed(question)
        if not q_emb:
            return self._json({"error": "Ollama unavailable"})
        hits = doc_db.search(q_emb, k)
        self._json({"contexts": [
            {"id": d.id, "title": d.title, "distance": round(dist, 4)}
            for dist, d in hits
        ]})

    def _doc_ask(self, body):
        question = body.get("question", "")
        k = parse_int(body.get("k"), 3)
        if not question:
            return self._json({"error": "need question"})

        # Step 1: embed the question
        q_emb = ollama.embed(question)
        if not q_emb:
            return self._json({"error": "Ollama unavailable"})

        # Step 2: retrieve top-k relevant chunks
        hits = doc_db.search(q_emb, k)

        # Step 3: build prompt
        ctx = "".join(f"[{i + 1}] {d.title}:\n{d.text}\n\n"
                      for i, (_, d) in enumerate(hits))
        prompt = (
            "You are a helpful assistant. Answer the user's question directly. "
            "Use the provided context if it contains relevant information. "
            "If it doesn't, just use your own general knowledge. "
            "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things "
            "like 'the context doesn't mention'. Just answer the question naturally.\n\n"
            f"Context:\n{ctx}Question: {question}\n\nAnswer:"
        )

        # Step 4: generate answer
        answer = ollama.generate(prompt)

        # Step 5: return everything
        self._json({
            "answer": answer,
            "model": ollama.gen_model,
            "contexts": [{"id": d.id, "title": d.title, "text": d.text,
                          "distance": round(dist, 4)} for dist, d in hits],
            "docCount": doc_db.size(),
        })

    def _status(self):
        self._json({
            "ollamaAvailable": ollama.is_available(),
            "embedModel": ollama.embed_model,
            "genModel": ollama.gen_model,
            "docCount": doc_db.size(),
            "docDims": doc_db.dims,
            "demoDims": DIMS,
            "demoCount": db.size(),
        })


# =====================================================================
#  MAIN
# =====================================================================

def main():
    load_demo(db)

    ollama_up = ollama.is_available()
    print("=== VectorDB Engine ===")
    print("http://localhost:8080")
    print(f"{db.size()} demo vectors | {DIMS} dims | HNSW+KD-Tree+BruteForce")
    print("Ollama:", "ONLINE" if ollama_up else "OFFLINE (install from ollama.com)")
    if ollama_up:
        print(f"  embed model: {ollama.embed_model}  gen model: {ollama.gen_model}")

    server = ThreadingHTTPServer(("0.0.0.0", 8080), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()