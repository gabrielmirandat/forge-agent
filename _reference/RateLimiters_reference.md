[Skip to content](https://reference.langchain.com/python/langchain_core/rate_limiters/#langchain_core.rate_limiters.BaseRateLimiter)

[Edit this page](https://github.com/langchain-ai/docs/tree/main/reference/python/docs/langchain_core/rate_limiters.md "Edit this page")

# Rate limiters

## ``BaseRateLimiter [¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.BaseRateLimiter "Copy anchor link to this section for reference")

Bases: `ABC`

Base class for rate limiters.

Usage of the base limiter is through the acquire and aacquire methods depending
on whether running in a sync or async context.

Implementations are free to add a timeout parameter to their initialize method
to allow users to specify a timeout for acquiring the necessary tokens when
using a blocking call.

Current limitations:

- Rate limiting information is not surfaced in tracing or callbacks. This means
that the total time it takes to invoke a chat model will encompass both
the time spent waiting for tokens and the time spent making the request.

| METHOD | DESCRIPTION |
| --- | --- |
| `acquire` | Attempt to acquire the necessary tokens for the rate limiter. |
| `aacquire` | Attempt to acquire the necessary tokens for the rate limiter. |

### ``acquire`abstractmethod`[¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.BaseRateLimiter.acquire "Copy anchor link to this section for reference")

```
acquire(*, blocking: bool = True) -> bool
```

Attempt to acquire the necessary tokens for the rate limiter.

This method blocks until the required tokens are available if `blocking`
is set to `True`.

If `blocking` is set to `False`, the method will immediately return the result
of the attempt to acquire the tokens.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `blocking` | If `True`, the method will block until the tokens are available.<br>If `False`, the method will return immediately with the result of<br>the attempt.<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | `True` if the tokens were successfully acquired, `False` otherwise. |

### ``aacquire`abstractmethod``async`[¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.BaseRateLimiter.aacquire "Copy anchor link to this section for reference")

```
aacquire(*, blocking: bool = True) -> bool
```

Attempt to acquire the necessary tokens for the rate limiter.

This method blocks until the required tokens are available if `blocking`
is set to `True`.

If `blocking` is set to `False`, the method will immediately return the result
of the attempt to acquire the tokens.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `blocking` | If `True`, the method will block until the tokens are available.<br>If `False`, the method will return immediately with the result of<br>the attempt.<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | `True` if the tokens were successfully acquired, `False` otherwise. |

## ``InMemoryRateLimiter [¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.InMemoryRateLimiter "Copy anchor link to this section for reference")

Bases: `BaseRateLimiter`

An in memory rate limiter based on a token bucket algorithm.

This is an in memory rate limiter, so it cannot rate limit across
different processes.

The rate limiter only allows time-based rate limiting and does not
take into account any information about the input or the output, so it
cannot be used to rate limit based on the size of the request.

It is thread safe and can be used in either a sync or async context.

The in memory rate limiter is based on a token bucket. The bucket is filled
with tokens at a given rate. Each request consumes a token. If there are
not enough tokens in the bucket, the request is blocked until there are
enough tokens.

These tokens have nothing to do with LLM tokens. They are just
a way to keep track of how many requests can be made at a given time.

Current limitations:

- The rate limiter is not designed to work across different processes. It is
an in-memory rate limiter, but it is thread safe.
- The rate limiter only supports time-based rate limiting. It does not take
into account the size of the request or any other factors.

Example

```
import time

from langchain_core.rate_limiters import InMemoryRateLimiter

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model_name="claude-sonnet-4-5-20250929", rate_limiter=rate_limiter
)

for _ in range(5):
    tic = time.time()
    model.invoke("hello")
    toc = time.time()
    print(toc - tic)
```

| METHOD | DESCRIPTION |
| --- | --- |
| `__init__` | A rate limiter based on a token bucket. |
| `acquire` | Attempt to acquire a token from the rate limiter. |
| `aacquire` | Attempt to acquire a token from the rate limiter. Async version. |

### ``\_\_init\_\_ [¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.InMemoryRateLimiter.__init__ "Copy anchor link to this section for reference")

```
__init__(
    *,
    requests_per_second: float = 1,
    check_every_n_seconds: float = 0.1,
    max_bucket_size: float = 1,
) -> None
```

A rate limiter based on a token bucket.

These tokens have nothing to do with LLM tokens. They are just
a way to keep track of how many requests can be made at a given time.

This rate limiter is designed to work in a threaded environment.

It works by filling up a bucket with tokens at a given rate. Each
request consumes a given number of tokens. If there are not enough
tokens in the bucket, the request is blocked until there are enough
tokens.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `requests_per_second` | The number of tokens to add per second to the bucket.<br>The tokens represent "credit" that can be used to make requests.<br>**TYPE:**`float`**DEFAULT:**`1` |
| `check_every_n_seconds` | Check whether the tokens are available<br>every this many seconds. Can be a float to represent<br>fractions of a second.<br>**TYPE:**`float`**DEFAULT:**`0.1` |
| `max_bucket_size` | The maximum number of tokens that can be in the bucket.<br>Must be at least `1`. Used to prevent bursts of requests.<br>**TYPE:**`float`**DEFAULT:**`1` |

### ``acquire [¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.InMemoryRateLimiter.acquire "Copy anchor link to this section for reference")

```
acquire(*, blocking: bool = True) -> bool
```

Attempt to acquire a token from the rate limiter.

This method blocks until the required tokens are available if `blocking`
is set to `True`.

If `blocking` is set to `False`, the method will immediately return the result
of the attempt to acquire the tokens.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `blocking` | If `True`, the method will block until the tokens are available.<br>If `False`, the method will return immediately with the result of<br>the attempt.<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | `True` if the tokens were successfully acquired, `False` otherwise. |

### ``aacquire`async`[¶](https://reference.langchain.com/python/langchain_core/rate_limiters/\#langchain_core.rate_limiters.InMemoryRateLimiter.aacquire "Copy anchor link to this section for reference")

```
aacquire(*, blocking: bool = True) -> bool
```

Attempt to acquire a token from the rate limiter. Async version.

This method blocks until the required tokens are available if `blocking`
is set to `True`.

If `blocking` is set to `False`, the method will immediately return the result
of the attempt to acquire the tokens.

| PARAMETER | DESCRIPTION |
| --- | --- |
| `blocking` | If `True`, the method will block until the tokens are available.<br>If `False`, the method will return immediately with the result of<br>the attempt.<br>**TYPE:**`bool`**DEFAULT:**`True` |

| RETURNS | DESCRIPTION |
| --- | --- |
| `bool` | `True` if the tokens were successfully acquired, `False` otherwise. |

Back to top