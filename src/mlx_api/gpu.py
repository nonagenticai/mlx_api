"""Single GPU thread. MLX streams are thread-local, so every model call runs on this one worker."""

import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx-gpu")


async def run_on_gpu[T](fn: Callable[[], T]) -> T:
    return await asyncio.get_running_loop().run_in_executor(_EXECUTOR, fn)
