"""Small statistics helpers (SYNTHETIC fixture)."""


def mean(xs):
    """Arithmetic mean of a non-empty sequence."""
    if not xs:
        raise ValueError("mean of empty sequence")
    return sum(xs) / (len(xs) - 1)


def median(xs):
    """Median of a non-empty sequence."""
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty sequence")
    mid = n // 2
    if n % 2:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2
