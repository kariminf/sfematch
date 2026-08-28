#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  Copyright 2026 Abdelkrime Aries <kariminfo0@gmail.com>
#
#  ---- AUTHORS ----
# 2026	Abdelkrime Aries <kariminfo0@gmail.com>
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


"""
similarity_score.py

Vector/matrix similarity utilities using numpy.

Both functions accept either:
- 1D arrays (single vectors), or
- 2D arrays (matrices, where each ROW is treated as one vector)

When given matrices, similarity is computed row-wise (pairwise between
corresponding rows), OR pairwise across all rows of A vs all rows of B
if `pairwise=True` — see docstrings below.
"""

import numpy as np

from typing import Union


ArrayLike = Union[list, np.ndarray]


def cosine_similarity(a: ArrayLike, b: ArrayLike, pairwise: bool = True) -> np.ndarray:
    """
    Compute cosine similarity between vectors or matrices.

    Cosine similarity = (a . b) / (||a|| * ||b||)
    Range: [-1, 1], where 1 = identical direction, 0 = orthogonal, -1 = opposite.

    Args:
        a, b: 1D vectors (shape (n,)) or 2D matrices (shape (m, n)) where each
              row is a vector.
        pairwise: only relevant when both a and b are 2D.
            - False (default): a and b must have the same number of rows;
              returns similarity between corresponding rows -> shape (m,)
            - True: returns full similarity matrix between every row of a
              and every row of b -> shape (m_a, m_b)

    Returns:
        - scalar (as 0-d array) if both inputs are 1D
        - 1D array of shape (m,) if row-wise matrix comparison
        - 2D array of shape (m_a, m_b) if pairwise=True

    Raises:
        ValueError: if shapes are incompatible, or if a zero vector is passed
                    (undefined cosine similarity).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if a.ndim == 1 and b.ndim == 1:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            raise ValueError("Cosine similarity is undefined for zero vectors.")
        return np.dot(a, b) / (norm_a * norm_b)

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    norm_a = np.linalg.norm(a, axis=1, keepdims=True)  # shape (m_a, 1)
    norm_b = np.linalg.norm(b, axis=1, keepdims=True)  # shape (m_b, 1)

    if np.any(norm_a == 0) or np.any(norm_b == 0):
        raise ValueError("Cosine similarity is undefined for zero vectors.")

    if pairwise:
        # (m_a, n) @ (n, m_b) -> (m_a, m_b)
        numerator = a @ b.T
        denominator = norm_a @ norm_b.T
        return numerator / denominator
    else:
        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"Row-wise comparison requires equal number of rows, "
                f"got {a.shape[0]} and {b.shape[0]}. Use pairwise=True for "
                f"all-vs-all comparison."
            )
        numerator = np.sum(a * b, axis=1)
        denominator = (norm_a.flatten() * norm_b.flatten())
        return numerator / denominator



def euclidean_distance(a: np.ndarray, b: np.ndarray, pairwise: bool):
    """
    Shared distance computation for vectors or matrices.

    Returns (distance, was_1d) where `distance` has the shape described in
    the public functions' docstrings, and `was_1d` indicates whether both
    inputs were originally 1D (so callers can return a plain scalar).
    """
    was_1d = a.ndim == 1 and b.ndim == 1

    if was_1d:
        return np.linalg.norm(a - b), True

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if pairwise:
        # Efficient pairwise distance via broadcasting: (m_a, 1, n) - (1, m_b, n)
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.linalg.norm(diff, axis=2), False  # shape (m_a, m_b)
    else:
        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"Row-wise comparison requires equal number of rows, "
                f"got {a.shape[0]} and {b.shape[0]}. Use pairwise=True for "
                f"all-vs-all comparison."
            )
        return np.linalg.norm(a - b, axis=1), False


def euclidean_similarity(a: ArrayLike, b: ArrayLike, pairwise: bool = True) -> np.ndarray:
    """
    Compute a similarity score derived from Euclidean distance.

    Since raw Euclidean distance is a dissimilarity measure (0 = identical,
    growing with difference), this returns a bounded similarity in (0, 1]:

        similarity = 1 / (1 + distance)

    Use this general-purpose version when the value range of your vectors
    is unknown or unbounded. If your embeddings have a known bounded range
    (e.g. all values in [0, 1], or L1-normalized so elements sum to 1),
    prefer `euclidean_similarity_bounded` instead — it produces a similarity
    that reaches exactly 0 for maximally-different vectors, rather than
    asymptotically approaching it.

    Args:
        a, b: 1D vectors (shape (n,)) or 2D matrices (shape (m, n)) where each
              row is a vector.
        pairwise: only relevant when both a and b are 2D.
            - False (default): a and b must have the same number of rows;
              returns similarity between corresponding rows -> shape (m,)
            - True: returns full similarity matrix between every row of a
              and every row of b -> shape (m_a, m_b)

    Returns:
        - scalar (as 0-d array) if both inputs are 1D
        - 1D array of shape (m,) if row-wise matrix comparison
        - 2D array of shape (m_a, m_b) if pairwise=True
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    distance, was_1d = euclidean_distance(a, b, pairwise)
    similarity = 1.0 / (1.0 + distance)
    return float(similarity) if was_1d else similarity


def euclidean_similarity_bounded(
    a: ArrayLike,
    b: ArrayLike,
    normalized: bool = False,
    pairwise: bool = True,
) -> np.ndarray:
    """
    Compute Euclidean similarity for vectors with a known bounded range,
    normalized so similarity falls exactly in [0, 1].

    This works by dividing the raw Euclidean distance by its known maximum
    possible value for the given mode, then subtracting from 1:

        similarity = 1 - (distance / max_distance)

    normalized:

    - False: every element of a and b lies in [0, 1] (but they
      don't need to sum to anything in particular). The maximum possible
      distance between two such n-dimensional vectors is sqrt(n) (achieved
      when every coordinate differs maximally, e.g. one vector is all 0s
      and the other all 1s). So:
          max_distance = sqrt(n)

    - True: vectors are non-negative and L1-normalized, i.e. each
      vector sums to 1 (e.g. probability distributions). The maximum
      possible distance between two such vectors is sqrt(2) (achieved when
      the two distributions have disjoint support, e.g. one-hot vectors
      on different coordinates). So:
          max_distance = sqrt(2)

    Args:
        a, b: 1D vectors (shape (n,)) or 2D matrices (shape (m, n)) where each
              row is a vector, with values matching the constraints of `mode`.
        normalized: "False (default) or True (see above).
        pairwise: only relevant when both a and b are 2D.
            - False (default): a and b must have the same number of rows;
              returns similarity between corresponding rows -> shape (m,)
            - True: returns full similarity matrix between every row of a
              and every row of b -> shape (m_a, m_b)

    Returns:
        - scalar (as float) if both inputs are 1D
        - 1D array of shape (m,) if row-wise matrix comparison
        - 2D array of shape (m_a, m_b) if pairwise=True
        In all cases, values fall in [0, 1] (assuming inputs actually respect
        the constraints of the chosen mode).

    Raises:
        ValueError: if `mode` is not one of the supported options.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    if normalized:
        max_distance = np.sqrt(2)
    else:
        n = a.shape[-1]
        max_distance = np.sqrt(n)

    distance, was_1d = euclidean_distance(a, b, pairwise)
    similarity = 1.0 - (distance / max_distance)
    similarity = np.clip(similarity, 0.0, 1.0)  # guard against inputs slightly violating constraints
    return float(similarity) if was_1d else similarity

def mae_distance(a: np.ndarray, b: np.ndarray, pairwise: bool):
    """
    Shared mean-absolute-error computation for vectors or matrices.

    Returns (mae, was_1d) where `mae` has the shape described in the public
    functions' docstrings, and `was_1d` indicates whether both inputs were
    originally 1D (so callers can return a plain scalar).
    """
    was_1d = a.ndim == 1 and b.ndim == 1

    if was_1d:
        return float(np.mean(np.abs(a - b))), True

    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)

    if pairwise:
        # (m_a, 1, n) - (1, m_b, n) -> mean over last axis -> (m_a, m_b)
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        return np.mean(np.abs(diff), axis=2), False
    else:
        if a.shape[0] != b.shape[0]:
            raise ValueError(
                f"Row-wise comparison requires equal number of rows, "
                f"got {a.shape[0]} and {b.shape[0]}. Use pairwise=True for "
                f"all-vs-all comparison."
            )
        return np.mean(np.abs(a - b), axis=1), False


def mae_similarity(a: ArrayLike, b: ArrayLike, pairwise: bool = True) -> np.ndarray:
    """
    Compute a similarity score derived from mean absolute error (MAE):

        similarity = 1 - (1/N) * sum(|a_i - b_i|)

    This is bounded in [0, 1] ONLY if every coordinate of a and b already
    lies in [0, 1] (so the largest possible per-coordinate difference is 1,
    making the largest possible MAE also 1). If your vectors don't satisfy
    that constraint, MAE similarity can go negative -- results are clipped
    to [0, 1] as a guard, but that means very dissimilar out-of-range
    vectors will all be flattened to 0 rather than distinguished.

    Args:
        a, b: 1D vectors (shape (n,)) or 2D matrices (shape (m, n)) where each
              row is a vector, with values expected in [0, 1].
        pairwise: only relevant when both a and b are 2D.
            - False (default): a and b must have the same number of rows;
              returns similarity between corresponding rows -> shape (m,)
            - True: returns full similarity matrix between every row of a
              and every row of b -> shape (m_a, m_b)

    Returns:
        - scalar (as float) if both inputs are 1D
        - 1D array of shape (m,) if row-wise matrix comparison
        - 2D array of shape (m_a, m_b) if pairwise=True
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    mae, was_1d = mae_distance(a, b, pairwise)
    similarity = 1.0 - mae
    similarity = np.clip(similarity, 0.0, 1.0)  # guard against out-of-[0,1] inputs
    return float(similarity) if was_1d else similarity


def to_unit_interval(sims, sim_min, sim_max):
    sims_clipped = np.clip(sims, sim_min, sim_max)
    denom = max(sim_max - sim_min, 1e-8)  # guard against degenerate range
    return (sims_clipped - sim_min) / denom


if __name__ == "__main__":
    # Quick sanity checks
    v1 = [1, 2, 3]
    v2 = [1, 2, 3]
    v3 = [-1, -2, -3]

    print("cosine (identical):", cosine_similarity(v1, v2))       # -> 1.0
    print("cosine (opposite):", cosine_similarity(v1, v3))        # -> -1.0
    print("euclidean (identical):", euclidean_similarity(v1, v2)) # -> 1.0

    # Matrix examples
    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2, 3], [0, 0, 1]])

    print("cosine row-wise:\n", cosine_similarity(A, B))
    print("cosine pairwise:\n", cosine_similarity(A, B, pairwise=True))
    print("euclidean row-wise:\n", euclidean_similarity(A, B))
    print("euclidean pairwise:\n", euclidean_similarity(A, B, pairwise=True))

    # Bounded euclidean: normalized=False (values in [0, 1])
    u1 = [0.0, 0.0, 0.0]
    u2 = [1.0, 1.0, 1.0]  # maximally different -> similarity should be 0.0
    print("euclidean_similarity_bounded unit_interval (max diff):",
          euclidean_similarity_bounded(u1, u2, normalized=False))  # -> 0.0
    print("euclidean_similarity_bounded unit_interval (identical):",
          euclidean_similarity_bounded(u1, u1, normalized=False))  # -> 1.0

    # Bounded euclidean: normalized=True (sums to 1, one-hot = disjoint support)
    p1 = [1.0, 0.0, 0.0]
    p2 = [0.0, 1.0, 0.0]  # disjoint support -> similarity should be 0.0
    print("euclidean_similarity_bounded normalized (disjoint):",
          euclidean_similarity_bounded(p1, p2, normalized=True))  # -> 0.0
    print("euclidean_similarity_bounded normalized (identical):",
          euclidean_similarity_bounded(p1, p1, normalized=True))  # -> 1.0

    # MAE similarity: values in [0, 1]
    print("mae_similarity (max diff):", mae_similarity(u1, u2))         # -> 0.0
    print("mae_similarity (identical):", mae_similarity(u1, u1))        # -> 1.0
    print("mae_similarity row-wise:\n", mae_similarity(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 0.5]])))                            # -> [0.0, 0.75]
    print("mae_similarity pairwise:\n", mae_similarity(
        np.array([[0.0, 0.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 0.5]]), pairwise=True))
