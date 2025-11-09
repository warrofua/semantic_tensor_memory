"""Test configuration for the Semantic Tensor Memory package."""
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import hashlib
import math
import pickle

import pytest


def _clone(data):
    if isinstance(data, list):
        return [_clone(x) for x in data]
    return data


def _shape(data):
    if isinstance(data, list):
        if not data:
            return (0,)
        return (len(data),) + _shape(data[0])
    return ()


def _flatten(data):
    if isinstance(data, list):
        result = []
        for item in data:
            result.extend(_flatten(item))
        return result
    return [data]


def _reshape(flat, shape):
    if not shape:
        return flat.pop(0)
    size = shape[0]
    return [_reshape(flat, shape[1:]) for _ in range(size)]


def _unsqueeze(data, dim):
    if dim == 0:
        return [_clone(data)]
    if isinstance(data, list):
        return [_unsqueeze(item, dim - 1) for item in data]
    raise ValueError("Unsupported unsqueeze dimension")


def _squeeze(data, dim=None):
    if dim is None or dim == 0:
        if isinstance(data, list) and len(data) == 1:
            return _squeeze(data[0], None if dim is None else 0)
        if isinstance(data, list):
            return [_squeeze(x, None) for x in data]
        return data
    if isinstance(data, list):
        return [_squeeze(x, dim - 1) for x in data]
    return data


def _mean_axis(data, axis):
    if axis == 0:
        length = len(data)
        if length == 0:
            return []
        cols = len(data[0]) if data and isinstance(data[0], list) else 0
        return [sum(row[c] for row in data) / length for c in range(cols)]
    if axis == 1:
        return [sum(row) / len(row) if row else 0.0 for row in data]
    raise NotImplementedError


def _mean_all(data):
    values = _flatten(data)
    return sum(values) / len(values) if values else 0.0


def _matmul(a, b):
    rows = len(a)
    cols = len(b[0]) if b and isinstance(b[0], list) else 0
    inner = len(a[0]) if a and isinstance(a[0], list) else 0
    result = []
    for i in range(rows):
        row = []
        for j in range(cols):
            total = 0.0
            for k in range(inner):
                total += a[i][k] * b[k][j]
            row.append(total)
        result.append(row)
    return result


def _apply_scalar(data, func):
    if isinstance(data, list):
        return [_apply_scalar(x, func) for x in data]
    return func(data)


def _cosine(vec_a, vec_b):
    numerator = sum(x * y for x, y in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(x * x for x in vec_a))
    norm_b = math.sqrt(sum(y * y for y in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return numerator / (norm_a * norm_b)


def _convert(data, dtype):
    if isinstance(data, list):
        return [_convert(x, dtype) for x in data]
    if dtype == "long":
        return int(data)
    return float(data)


def _index(data, key):
    if isinstance(key, slice):
        return data[key]
    if isinstance(key, (list, tuple)):
        return [data[k] for k in key]
    return data[key]


class _Tensor:
    """Minimal tensor wrapper built on Python lists."""

    def __init__(self, data, dtype="float"):
        if isinstance(data, _Tensor):
            data = data._data
            dtype = data._dtype
        self._data = _clone(data)
        self._dtype = dtype

    def unsqueeze(self, dim: int):
        return _Tensor(_unsqueeze(self._data, dim), self._dtype)

    def squeeze(self, dim=None):
        return _Tensor(_squeeze(self._data, dim), self._dtype)

    def reshape(self, *shape):
        flat = _flatten(self._data)
        reshaped = _reshape(flat, list(shape))
        return _Tensor(reshaped, self._dtype)

    def mean(self, dim=None):
        if dim is None:
            return _Tensor(_mean_all(self._data), "float")
        data = self.tolist()
        if not isinstance(data, list) or not data:
            return _Tensor(0.0, "float")
        return _Tensor(_mean_axis(data, dim), "float")

    def numpy(self):
        return self

    def tolist(self):
        return _clone(self._data)

    def item(self):
        value = self._data
        while isinstance(value, list):
            if not value:
                return 0.0
            value = value[0]
        return value

    def __iter__(self):
        if not isinstance(self._data, list):
            raise TypeError("Scalar tensor is not iterable")
        for item in self._data:
            yield _Tensor(item, self._dtype if isinstance(item, list) else "float")

    def detach(self):
        return _Tensor(self._data, self._dtype)

    def clone(self):
        return _Tensor(_clone(self._data), self._dtype)

    def to(self, *args, **kwargs):
        return self

    def __len__(self):
        if isinstance(self._data, list):
            return len(self._data)
        return 1

    @property
    def shape(self):
        return _shape(self._data)

    def __getitem__(self, key):
        data = self._data
        if isinstance(key, tuple):
            for part in key:
                data = _index(data, part)
        else:
            data = _index(data, key)
        return _Tensor(data, self._dtype if isinstance(data, list) else "float") if isinstance(data, list) else data

    def __matmul__(self, other):
        return _Tensor(_matmul(self.tolist(), other.tolist()), "float")

    def t(self):
        data = self.tolist()
        transposed = [list(row) for row in zip(*data)]
        return _Tensor(transposed, "float")

    def __rsub__(self, other):
        return _Tensor(_apply_scalar(self._data, lambda x: other - x), "float")

    def __repr__(self):
        return f"Tensor({self._data!r})"

    def dim(self):
        return len(self.shape)

# Stub torch module
if "torch" not in sys.modules:
    torch_stub = ModuleType("torch")
    torch_stub.Tensor = _Tensor
    torch_stub.float32 = "float"
    torch_stub.long = "long"

    class _Device(str):
        pass

    def _device(spec):
        return _Device(str(spec))

    def _tensor(data, dtype=None):
        dtype_label = "long" if dtype == torch_stub.long else "float"
        return _Tensor(_convert(data, dtype_label), dtype_label)

    def _arange(end, dtype=None):
        dtype_label = "long" if dtype == torch_stub.long else "float"
        values = [int(i) if dtype_label == "long" else float(i) for i in range(end)]
        return _Tensor(values, dtype_label)

    def _stack(tensors, dim=0):
        lists = [t.tolist() for t in tensors]
        if dim == 0:
            return _Tensor(lists, "float")
        raise NotImplementedError

    def _cosine_similarity_tensor(a, b, dim=0):
        data_a = a.tolist()
        data_b = b.tolist()
        if isinstance(data_a[0], list):
            if dim == 1:
                result = [_cosine(row_a, row_b) for row_a, row_b in zip(data_a, data_b)]
                return _Tensor(result, "float")
            if dim == 0:
                columns_a = list(zip(*data_a))
                columns_b = list(zip(*data_b))
                result = [_cosine(list(col_a), list(col_b)) for col_a, col_b in zip(columns_a, columns_b)]
                return _Tensor(result, "float")
        return _Tensor(_cosine(data_a, data_b), "float")

    def _normalize_tensor(tensor, p=2, dim=1):
        data = tensor.tolist()
        if dim == 1:
            normalized = []
            for row in data:
                norm = math.sqrt(sum(x * x for x in row))
                norm = norm if norm else 1.0
                normalized.append([x / norm for x in row])
            return _Tensor(normalized, "float")
        if dim == 0:
            columns = list(zip(*data))
            normalized_cols = []
            for col in columns:
                norm = math.sqrt(sum(x * x for x in col))
                norm = norm if norm else 1.0
                normalized_cols.append([x / norm for x in col])
            normalized = [list(row) for row in zip(*normalized_cols)]
            return _Tensor(normalized, "float")
        raise NotImplementedError

    def _allclose(a, b, atol=1e-8, rtol=1e-5):
        vals_a = _flatten(a.tolist())
        vals_b = _flatten(b.tolist())
        return all(abs(x - y) <= atol + rtol * abs(y) for x, y in zip(vals_a, vals_b))

    def _ones(shape, dtype=None):
        dtype_label = "long" if dtype == torch_stub.long else "float"
        if isinstance(shape, tuple) and len(shape) == 2:
            data = [[1 if dtype_label == "long" else 1.0 for _ in range(shape[1])] for _ in range(shape[0])]
        else:
            size = shape if isinstance(shape, int) else shape[0]
            data = [1 if dtype_label == "long" else 1.0 for _ in range(size)]
        return _Tensor(data, dtype_label)

    def _triu(tensor, diagonal=0):
        data = tensor.tolist()
        size = len(data)
        result = []
        for i in range(size):
            row = []
            for j in range(len(data[i])):
                if j - i >= diagonal:
                    row.append(data[i][j])
                else:
                    row.append(0.0)
            result.append(row)
        return _Tensor(result, "float")

    def _mm(a, b):
        return _Tensor(_matmul(a.tolist(), b.tolist()), "float")

    torch_stub.tensor = _tensor
    torch_stub.arange = _arange
    torch_stub.stack = _stack
    torch_stub.cosine_similarity = _cosine_similarity_tensor
    torch_stub.allclose = _allclose
    torch_stub.ones = _ones
    torch_stub.triu = _triu
    torch_stub.mm = _mm
    torch_stub.device = _device

    def _save(obj, path):
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_bytes(pickle.dumps(obj))

    torch_stub.save = _save

    def _load(path, map_location=None):
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(path)
        return pickle.loads(file_path.read_bytes())

    torch_stub.load = _load

    def _inference_mode(func=None):
        if func is None:
            def wrapper(f):
                return f
            return wrapper
        return func

    torch_stub.inference_mode = _inference_mode

    nn_module = ModuleType("nn")
    functional_module = ModuleType("functional")
    functional_module.normalize = _normalize_tensor
    functional_module.cosine_similarity = _cosine_similarity_tensor
    nn_module.functional = functional_module
    torch_stub.nn = nn_module

    testing_module = ModuleType("testing")

    def _assert_close(actual, expected, rtol=1e-5, atol=1e-8):
        if not _allclose(actual, expected, atol=atol, rtol=rtol):
            raise AssertionError("Tensors are not close")

    testing_module.assert_close = _assert_close
    torch_stub.testing = testing_module

    sys.modules["torch"] = torch_stub

import torch

# Stub drift module with deterministic implementations
if "semantic_tensor_memory.memory.drift" not in sys.modules:
    drift_module = ModuleType("semantic_tensor_memory.memory.drift")

    def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
        vec_a = _flatten(a.tolist())
        vec_b = _flatten(b.tolist())
        return _cosine(vec_a, vec_b)

    def session_mean(mat: torch.Tensor) -> torch.Tensor:
        data = mat.tolist()
        if len(data) == 1:
            return torch.tensor(data[0])
        return torch.tensor(_mean_axis(data, 0))

    def drift_series(tensors):
        means = [session_mean(t) for t in tensors]
        token_counts = [t.shape[0] if isinstance(t.shape, tuple) else 1 for t in tensors]
        drifts = []
        for i in range(1, len(means)):
            sim = torch.nn.functional.cosine_similarity(means[i], means[i - 1], dim=0).item()
            drifts.append(1 - sim)
        return drifts, token_counts

    def token_drift(tensors, window=3):
        return []

    drift_module.cosine = cosine
    drift_module.session_mean = session_mean
    drift_module.drift_series = drift_series
    drift_module.token_drift = token_drift
    sys.modules["semantic_tensor_memory.memory.drift"] = drift_module

# Stub sequence_drift module
if "semantic_tensor_memory.memory.sequence_drift" not in sys.modules:
    seq_module = ModuleType("semantic_tensor_memory.memory.sequence_drift")

    def sequence_drift(tensors, max_length=32):
        scores = []
        for i in range(1, len(tensors)):
            prev_vectors = tensors[i - 1].tolist()[:max_length]
            curr_vectors = tensors[i].tolist()[:max_length]
            if not prev_vectors or not curr_vectors:
                scores.append(1.0)
                continue
            prev_mean = _mean_axis(prev_vectors, 0)
            curr_mean = _mean_axis(curr_vectors, 0)
            sim = _cosine(prev_mean, curr_mean)
            scores.append(1 - sim)
        return scores

    def semantic_coherence_score(tensor):
        vectors = tensor.tolist()
        if len(vectors) < 2:
            return 1.0
        total = 0.0
        count = 0
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                total += _cosine(vectors[i], vectors[j])
                count += 1
        return total / count if count else 1.0

    def token_importance_drift(tensors, top_k=10):
        return []

    def enhanced_drift_analysis(tensors, meta):
        return {
            "sequence_drift": sequence_drift(tensors),
            "token_importance": token_importance_drift(tensors),
            "coherence_scores": [semantic_coherence_score(t) for t in tensors],
            "session_lengths": [t.shape[0] for t in tensors],
        }

    seq_module.sequence_drift = sequence_drift
    seq_module.semantic_coherence_score = semantic_coherence_score
    seq_module.token_importance_drift = token_importance_drift
    seq_module.enhanced_drift_analysis = enhanced_drift_analysis
    sys.modules["semantic_tensor_memory.memory.sequence_drift"] = seq_module

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def stub_embeddings(monkeypatch):
    """Provide deterministic embedding functions without loading external models."""

    dim = 4

    class _DummyTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, padding=False):
            tokens = text.split()
            length = len(tokens) + 2
            input_ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
            return {"input_ids": input_ids}

        def convert_ids_to_tokens(self, ids):
            return [f"tok{i}" for i in ids.tolist()]

    class _DummyModel:
        def __init__(self):
            self.config = SimpleNamespace(hidden_size=dim)

        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            seq_len = input_ids.shape[1]
            values = [[float(i * dim + j) for j in range(dim)] for i in range(seq_len)]
            tensor = torch.tensor(values).unsqueeze(0)
            return SimpleNamespace(last_hidden_state=tensor)

    class _DummySentenceModel:
        def __init__(self, *args, **kwargs):
            self._dim = dim

        def get_sentence_embedding_dimension(self):
            return self._dim

        def encode(self, text, convert_to_tensor=True, show_progress_bar=False):
            return _sentence_vector(text)

    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: _DummyTokenizer(),
    )
    monkeypatch.setattr(
        "transformers.AutoModel.from_pretrained",
        lambda *args, **kwargs: _DummyModel(),
    )
    monkeypatch.setattr(
        "sentence_transformers.SentenceTransformer",
        _DummySentenceModel,
    )

    embedder = importlib.import_module("semantic_tensor_memory.memory.embedder")
    dual_embedder = importlib.import_module("semantic_tensor_memory.memory.dual_embedder")
    embedder_config = importlib.import_module("semantic_tensor_memory.memory.embedder_config")

    importlib.reload(embedder)
    importlib.reload(dual_embedder)
    importlib.reload(embedder_config)

    def _digest(key: str) -> torch.Tensor:
        data = hashlib.sha256(key.encode("utf-8")).digest()
        values = [data[i] / 255.0 for i in range(dim)]
        return torch.tensor(values)

    def _tokenize(text: str):
        tokens = [tok for tok in text.lower().split() if tok]
        if not tokens:
            tokens = [""]
        return tokens

    def _sentence_vector(text: str) -> torch.Tensor:
        return _digest(f"sentence::{text}")

    def _token_matrix(text: str) -> torch.Tensor:
        tokens = _tokenize(text)
        vectors = [_digest(f"token::{text}::{idx}::{tok}") for idx, tok in enumerate(tokens)]
        return torch.stack(vectors), tokens

    def _embed_sentence(text: str) -> torch.Tensor:
        return _sentence_vector(text).unsqueeze(0)

    def _create_dual_embedding(text: str):
        token_embeddings, tokens = _token_matrix(text)
        return dual_embedder.DualEmbedding(
            token_embeddings=token_embeddings,
            sentence_embedding=_sentence_vector(text),
            text=text,
            token_count=len(tokens),
            tokens=tokens,
        )

    monkeypatch.setattr(embedder, "embed_sentence", _embed_sentence)
    monkeypatch.setattr(dual_embedder, "create_dual_embedding", _create_dual_embedding)
    monkeypatch.setattr(dual_embedder, "embed_sentence", lambda text: _create_dual_embedding(text).token_embeddings)
    monkeypatch.setattr(dual_embedder, "get_token_count", lambda text: len(_tokenize(text)))

    def _sentence_similarity(text_a: str, text_b: str) -> float:
        vec_a = _sentence_vector(text_a)
        vec_b = _sentence_vector(text_b)
        return torch.nn.functional.cosine_similarity(vec_a, vec_b, dim=0).item()

    def _token_embeddings(text: str) -> torch.Tensor:
        return _token_matrix(text)[0]

    def _tokens(text: str):
        return _token_matrix(text)[1]

    def _token_mean_similarity(text_a: str, text_b: str) -> float:
        emb_a = _token_embeddings(text_a).mean(0)
        emb_b = _token_embeddings(text_b).mean(0)
        return torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()

    return SimpleNamespace(
        embedder=embedder,
        dual_embedder=dual_embedder,
        embedder_config=embedder_config,
        dim=dim,
        create_dual_embedding=_create_dual_embedding,
        embed_sentence=_embed_sentence,
        sentence_vector=_sentence_vector,
        token_embeddings=_token_embeddings,
        tokens=_tokens,
        sentence_similarity=_sentence_similarity,
        token_mean_similarity=_token_mean_similarity,
    )

# Lightweight stubs for external transformer libraries
if "numpy" not in sys.modules:
    numpy_stub = ModuleType("numpy")

    def _to_list(values):
        if hasattr(values, "tolist"):
            return values.tolist()
        if isinstance(values, list):
            return values
        return list(values)

    def _mean(values):
        data = _to_list(values)
        return sum(data) / len(data) if data else 0.0

    def array(values):
        return _to_list(values)

    def asarray(values):
        return _to_list(values)

    def mean(values, axis=None):
        if axis is not None:
            return _mean(values)
        return _mean(values)

    def triu_indices_from(matrix, k=0):
        data = _to_list(matrix)
        size = len(data)
        rows, cols = [], []
        for i in range(size):
            for j in range(i + max(k, 1), size):
                rows.append(i)
                cols.append(j)
        return rows, cols

    def where(condition):
        indices = [idx for idx, flag in enumerate(condition) if flag]
        return (indices,)

    numpy_stub.array = array
    numpy_stub.asarray = asarray
    numpy_stub.mean = mean
    numpy_stub.triu_indices_from = triu_indices_from
    numpy_stub.where = where
    numpy_stub.ndarray = list

    sys.modules["numpy"] = numpy_stub

if "sklearn" not in sys.modules:
    sklearn_stub = ModuleType("sklearn")
    cluster_module = ModuleType("cluster")
    manifold_module = ModuleType("manifold")
    metrics_module = ModuleType("metrics")
    pairwise_module = ModuleType("pairwise")

    class _StubKMeans:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("KMeans stub requires numpy/scikit-learn")

    class _StubTSNE:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("TSNE stub requires numpy/scikit-learn")

    def _cosine_similarity(a, b=None):
        raise RuntimeError("cosine_similarity stub requires numpy/scikit-learn")

    cluster_module.KMeans = _StubKMeans
    manifold_module.TSNE = _StubTSNE
    pairwise_module.cosine_similarity = _cosine_similarity
    metrics_module.pairwise = pairwise_module
    sklearn_stub.cluster = cluster_module
    sklearn_stub.manifold = manifold_module
    sklearn_stub.metrics = metrics_module
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.cluster"] = cluster_module
    sys.modules["sklearn.manifold"] = manifold_module
    sys.modules["sklearn.metrics"] = metrics_module
    sys.modules["sklearn.metrics.pairwise"] = pairwise_module

if "pandas" not in sys.modules:
    pandas_stub = ModuleType("pandas")

    class _DataFrame:
        def __init__(self, *args, **kwargs):
            self._data = args[0] if args else {}

        def to_dict(self):
            return self._data

    pandas_stub.DataFrame = _DataFrame
    pandas_stub.Series = list

    sys.modules["pandas"] = pandas_stub

if "transformers" not in sys.modules:
    transformers_stub = ModuleType("transformers")

    class _AutoTokenizer:
        def __call__(self, text, return_tensors="pt", truncation=True, padding=False):
            tokens = str(text).split()
            length = len(tokens) + 2
            input_ids = torch.arange(length, dtype=torch.long).unsqueeze(0)
            return {"input_ids": input_ids}

        def convert_ids_to_tokens(self, ids):
            flat_ids = ids.view(-1) if hasattr(ids, "view") else ids
            return [f"tok{i}" for i in flat_ids.tolist()]

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class _AutoModel:
        def __init__(self):
            self.config = SimpleNamespace(hidden_size=4)

        def __call__(self, **kwargs):
            input_ids = kwargs.get("input_ids")
            if hasattr(input_ids, "shape"):
                seq_len = input_ids.shape[1]
            else:
                seq_len = len(input_ids[0]) if input_ids else 0
            dim = self.config.hidden_size
            values = [[float(i * dim + j) for j in range(dim)] for i in range(seq_len)]
            tensor = torch.tensor(values).unsqueeze(0)
            return SimpleNamespace(last_hidden_state=tensor)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.AutoModel = _AutoModel
    sys.modules["transformers"] = transformers_stub

if "sentence_transformers" not in sys.modules:
    sentence_stub = ModuleType("sentence_transformers")

    class _SentenceTransformer:
        def __init__(self, *args, **kwargs):
            self._dim = 4

        def get_sentence_embedding_dimension(self):
            return self._dim

        def encode(self, text, convert_to_tensor=True, show_progress_bar=False):
            vec = torch.arange(self._dim, dtype=torch.float)
            return vec if convert_to_tensor else vec.tolist()

    sentence_stub.SentenceTransformer = _SentenceTransformer
    sys.modules["sentence_transformers"] = sentence_stub

    def _inference_mode(func=None):
        if func is None:
            def wrapper(f):
                return f
            return wrapper
        return func

    torch_stub.inference_mode = _inference_mode

# Minimal stub for the storage module to avoid optional dependencies
if "semantic_tensor_memory.memory.store" not in sys.modules:
    store_stub = ModuleType("semantic_tensor_memory.memory.store")

    def _noop(*args, **kwargs):
        return None

    def _load(*args, **kwargs):
        return []

    store_stub.load = _load
    store_stub.save = _noop
    store_stub.append = _noop
    store_stub.to_batch = lambda *args, **kwargs: []
    store_stub.flatten = lambda *args, **kwargs: []
    sys.modules["semantic_tensor_memory.memory.store"] = store_stub
