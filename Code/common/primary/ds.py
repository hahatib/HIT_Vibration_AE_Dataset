"""Normalized Dempster rule with pignistic decision probabilities."""

from dataclasses import dataclass
import numpy as np

NUMERICAL_EPSILON = 1e-12


def _probability_matrix(probabilities, name="probabilities") -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] < 2:
        raise ValueError(
            f"{name} must have shape (samples, classes), found {values.shape}."
        )
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError(f"{name} contains negative or non-finite values.")
    row_sums = values.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{name} contains a row with zero total probability.")
    return values / row_sums


def probability_nll(probabilities, labels) -> float:
    values = _probability_matrix(probabilities)
    labels = _label_vector(labels, len(values), values.shape[1])
    true_values = values[np.arange(len(labels)), labels]
    return float(-np.mean(np.log(np.clip(true_values, NUMERICAL_EPSILON, 1.0))))


@dataclass(frozen=True)
class DiscountedBPA:
    singletons: np.ndarray
    ignorance: np.ndarray


@dataclass(frozen=True)
class CombinationResult:
    probabilities: np.ndarray
    predictions: np.ndarray
    conflict: np.ndarray
    singleton_masses: np.ndarray
    ignorance_mass: np.ndarray
    fallback_mask: np.ndarray


def build_discounted_bpa(probabilities, reliability) -> DiscountedBPA:
    values = _probability_matrix(probabilities)
    reliability = np.asarray(reliability, dtype=np.float64)
    if reliability.ndim == 0:
        reliability = np.full(len(values), float(reliability), dtype=np.float64)
    elif reliability.shape == (len(values), 1):
        reliability = reliability[:, 0]
    if reliability.shape != (len(values),):
        raise ValueError(
            f"reliability must be scalar or shape ({len(values)},), found {reliability.shape}."
        )
    if (
        not np.all(np.isfinite(reliability))
        or np.any(reliability < 0.0)
        or np.any(reliability > 1.0)
    ):
        raise ValueError("reliability must be finite and lie in [0, 1].")
    singletons = reliability[:, None] * values
    ignorance = 1.0 - singletons.sum(axis=1)
    return DiscountedBPA(singletons=singletons, ignorance=ignorance)


def _validate_bpa(bpa: DiscountedBPA, name: str) -> DiscountedBPA:
    singletons = np.asarray(bpa.singletons, dtype=np.float64)
    ignorance = np.asarray(bpa.ignorance, dtype=np.float64)
    if singletons.ndim != 2 or ignorance.shape != (len(singletons),):
        raise ValueError(f"{name} has incompatible singleton and ignorance shapes.")
    if not np.all(np.isfinite(singletons)) or not np.all(np.isfinite(ignorance)):
        raise ValueError(f"{name} contains non-finite masses.")
    if np.any(singletons < -NUMERICAL_EPSILON) or np.any(
        ignorance < -NUMERICAL_EPSILON
    ):
        raise ValueError(f"{name} contains negative masses.")
    totals = singletons.sum(axis=1) + ignorance
    if not np.allclose(totals, 1.0, atol=1e-09, rtol=0.0):
        raise ValueError(f"{name} masses do not sum to one.")
    return DiscountedBPA(np.clip(singletons, 0.0, 1.0), np.clip(ignorance, 0.0, 1.0))


def _source_probabilities(bpa: DiscountedBPA) -> np.ndarray:
    strength = bpa.singletons.sum(axis=1, keepdims=True)
    class_count = bpa.singletons.shape[1]
    return np.divide(
        bpa.singletons,
        strength,
        out=np.full_like(bpa.singletons, 1.0 / class_count),
        where=strength > NUMERICAL_EPSILON,
    )


def _fallback_probabilities(first: DiscountedBPA, second: DiscountedBPA) -> np.ndarray:
    probability_first = _source_probabilities(first)
    probability_second = _source_probabilities(second)
    strength_first = first.singletons.sum(axis=1)
    strength_second = second.singletons.sum(axis=1)
    choose_first = strength_first > strength_second + NUMERICAL_EPSILON
    choose_second = strength_second > strength_first + NUMERICAL_EPSILON
    fallback = 0.5 * (probability_first + probability_second)
    fallback[choose_first] = probability_first[choose_first]
    fallback[choose_second] = probability_second[choose_second]
    return fallback / fallback.sum(axis=1, keepdims=True)


def combine_dempster(
    first: DiscountedBPA, second: DiscountedBPA, *, epsilon=NUMERICAL_EPSILON
) -> CombinationResult:
    first = _validate_bpa(first, "first")
    second = _validate_bpa(second, "second")
    if first.singletons.shape != second.singletons.shape:
        raise ValueError("The two BPAs have different shapes.")
    product_same = first.singletons * second.singletons
    conflict = first.singletons.sum(axis=1) * second.singletons.sum(
        axis=1
    ) - product_same.sum(axis=1)
    denominator = 1.0 - conflict
    fallback_mask = (
        ~np.isfinite(denominator)
        | (denominator <= float(epsilon))
        | ~np.isfinite(conflict)
    )
    safe_denominator = np.where(fallback_mask, 1.0, denominator)
    conjunctive_singletons = (
        product_same
        + first.singletons * second.ignorance[:, None]
        + first.ignorance[:, None] * second.singletons
    )
    singleton_masses = conjunctive_singletons / safe_denominator[:, None]
    ignorance_mass = first.ignorance * second.ignorance / safe_denominator
    class_count = first.singletons.shape[1]
    probabilities = singleton_masses + ignorance_mass[:, None] / class_count
    invalid_probability = ~np.all(np.isfinite(probabilities), axis=1) | (
        probabilities.sum(axis=1) <= 0.0
    )
    fallback_mask = fallback_mask | invalid_probability
    if np.any(fallback_mask):
        probabilities[fallback_mask] = _fallback_probabilities(first, second)[
            fallback_mask
        ]
        singleton_masses[fallback_mask] = probabilities[fallback_mask]
        ignorance_mass[fallback_mask] = 0.0
    probabilities = np.clip(probabilities, 0.0, None)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return CombinationResult(
        probabilities=probabilities,
        predictions=probabilities.argmax(axis=1),
        conflict=np.clip(conflict, 0.0, 1.0),
        singleton_masses=singleton_masses,
        ignorance_mass=ignorance_mass,
        fallback_mask=fallback_mask,
    )


def _label_vector(labels, sample_count, class_count) -> np.ndarray:
    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or len(values) != sample_count:
        raise ValueError(
            f"labels must have shape ({sample_count},), found {values.shape}."
        )
    if np.any(values < 0) or np.any(values >= class_count):
        raise ValueError("labels contain an index outside the class range.")
    return values
