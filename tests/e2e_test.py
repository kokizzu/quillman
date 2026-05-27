# For CI smoke testing only
# Assumes serve endpoint running at modal-labs--quillman-moshi-web-dev.modal.run

import asyncio
from pathlib import Path

import os
import aiohttp
import time

endpoint = "wss://modal-labs--quillman-moshi-web-dev.modal.run/"

shutdown_flag = asyncio.Event()

WARMUP_TIMEOUT = 180 # cold boot can include HF download + GPU warmup
async def ensure_server_ready():
    deadline = time.time() + WARMUP_TIMEOUT
    while time.time() < deadline:
        try:
            print("Probing server with ws_connect...")
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(
                    endpoint + "ws",
                    timeout=aiohttp.ClientTimeout(total=60),
                ):
                    print("Server ready.")
                    return
        except Exception as e:
            print("Server not ready yet:", e)
            await asyncio.sleep(5)
    raise RuntimeError(f"Server did not become ready within {WARMUP_TIMEOUT}s")

async def run():
    await ensure_server_ready()

    send_chunks = []
    recv_chunks = []

    files = os.listdir(Path(__file__).parent / "e2e_in")
    files.sort()
    for chunk_file in files:
        with open(Path(__file__).parent / "e2e_in" / chunk_file, "rb") as f:
            data = f.read()
            send_chunks.append(data)

    print("Connecting to", endpoint + "ws")
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(endpoint + "ws") as ws:
            print("Connection established.")
            
            async def send_loop():
                for chunk in send_chunks:
                    await asyncio.sleep(0.1)
                    await ws.send_bytes(chunk)
                    print("Sent chunk, len:", len(chunk))

            async def recv_loop():
                async for msg in ws:
                    if shutdown_flag.is_set():
                        break
                    data = msg.data
                    if not isinstance(data, bytes) or len(data) == 0:
                        continue
                    print("Received chunk, len:", len(data))
                    recv_chunks.append(data)

            async def timeout_loop():
                await asyncio.sleep(10)
                shutdown_flag.set()
                await ws.close()
        
            await asyncio.gather(send_loop(), recv_loop(), timeout_loop())

    assert(len(recv_chunks) >= 3) # Opus sends two headers always, so at least three should be received for healthy inference
    print("Done")

if __name__ == "__main__":
    asyncio.run(run())