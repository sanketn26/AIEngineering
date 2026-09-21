"""Bounded HTTP load probe; reports failures alongside latency."""

import argparse
import asyncio
import json
import math
import os
import time
import httpx


async def run(url: str, n: int, concurrency: int):
    semaphore = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=10) as client:

        async def one(i):
            async with semaphore:
                start = time.perf_counter()
                try:
                    response = await client.post(
                        url.rstrip("/") + "/v1/triage",
                        headers={
                            "Authorization": "Bearer " + os.environ["TRIAGE_BEARER"]
                        },
                        json={"ticket_id": f"load-{i}", "text": "I forgot my password"},
                    )
                    status = response.status_code
                except httpx.HTTPError:
                    status = 0
                return status, (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        rows = await asyncio.gather(*(one(i) for i in range(n)))
    times = sorted(t for _, t in rows)
    elapsed = time.perf_counter() - start
    return {
        "n": n,
        "concurrency": concurrency,
        "successes": sum(s == 200 for s, _ in rows),
        "errors": sum(s != 200 for s, _ in rows),
        "elapsed_s": elapsed,
        "successful_requests_per_s": sum(s == 200 for s, _ in rows) / elapsed,
        "latency_includes_failures": True,
        **{f"p{int(p*100)}_ms": times[math.ceil(p * n) - 1] for p in (0.5, 0.95, 0.99)},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.requests <= 1000 or not 1 <= args.concurrency <= 32:
        parser.error("use 1–1000 requests and 1–32 concurrent requests")
    print(
        json.dumps(
            asyncio.run(run(args.url, args.requests, args.concurrency)), indent=2
        )
    )
