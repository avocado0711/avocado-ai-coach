import pytest

from app.rate_limit import RateLimitExceeded, SlidingWindowLimiter


def test_limiter_blocks_after_limit():
    limiter = SlidingWindowLimiter()
    limiter.check("user", 2, 60)
    limiter.check("user", 2, 60)
    with pytest.raises(RateLimitExceeded):
        limiter.check("user", 2, 60)


def test_limiter_isolated_by_key():
    limiter = SlidingWindowLimiter()
    limiter.check("user-a", 1, 60)
    limiter.check("user-b", 1, 60)

