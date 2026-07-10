"""
Proxy connectivity test script.

Reads PROXY_URL from .env and checks connectivity to:
  1. External hosts via direct TCP
  2. Backend API via proxy
  3. httpbin via proxy (IP/location check, requires --geo)

Usage:
  python check_proxy.py          # basic checks
  python check_proxy.py --geo    # + geo check via httpbin
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx
from bot.config import get_bot_settings

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> str:
    return f"{GREEN}[OK]{RESET} {msg}"


def fail(msg: str) -> str:
    return f"{RED}[FAIL]{RESET} {msg}"


def info(msg: str) -> str:
    return f"{CYAN}[...]{RESET} {msg}"


async def check_tcp(host: str, port: int, timeout: float = 5.0) -> tuple[bool, str]:
    """Direct TCP connectivity check (no proxy)."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=timeout
        )
        writer.close()
        await writer.wait_closed()
        return True, f"TCP {host}:{port} OK"
    except asyncio.TimeoutError:
        return False, f"TCP {host}:{port} timeout ({timeout}s)"
    except OSError as e:
        return False, f"TCP {host}:{port} {e}"


async def check_http_via_proxy(
    proxy_url: str, target_url: str, timeout: float = 10.0
) -> tuple[bool, str, str | None]:
    """HTTP request through proxy."""
    try:
        async with httpx.AsyncClient(
            proxy=proxy_url,
            timeout=httpx.Timeout(timeout),
            follow_redirects=True,
        ) as client:
            resp = await client.get(target_url)
            body_preview = resp.text[:200].replace("\n", " ")
            return (
                resp.status_code < 500,
                f"{target_url} -> HTTP {resp.status_code}",
                body_preview,
            )
    except httpx.ConnectError as e:
        return False, f"{target_url} connection refused: {e}", None
    except httpx.ReadTimeout:
        return False, f"{target_url} read timeout ({timeout}s)", None
    except httpx.ConnectTimeout:
        return False, f"{target_url} connect timeout ({timeout}s)", None
    except Exception as e:
        return False, f"{target_url} {type(e).__name__}: {e}", None


async def main() -> None:
    settings = get_bot_settings()
    proxy_url = settings.proxy_url
    backend_url = settings.backend_url

    print(f"\n{BOLD}=== Proxy Check ==={RESET}\n")

    # Config
    print(f"PROXY_URL  = {YELLOW}{proxy_url or '(not set)'}{RESET}")
    print(f"BACKEND_URL = {backend_url or '(not set)'}")
    print(f"BOT_TOKEN  = {settings.bot_token.get_secret_value()[:12]}...")
    print()

    if not proxy_url:
        print(fail("PROXY_URL is not set in .env"))
        print(info("Add PROXY_URL=http://user:pass@host:port to .env"))
        return

    # Direct TCP (no proxy - basic internet check)
    print(f"{BOLD}--- Direct TCP ---{RESET}")
    for host, port in [("api.telegram.org", 443), ("google.com", 443)]:
        success, msg = await check_tcp(host, port, timeout=5.0)
        print(ok(msg) if success else fail(msg))
    print()

    # HTTP via proxy
    print(f"{BOLD}--- HTTP via proxy ---{RESET}")

    # Backend checks
    if backend_url:
        for path in ["/health", "/ready"]:
            success, msg, body = await check_http_via_proxy(
                proxy_url, f"{backend_url}{path}"
            )
            print(ok(msg) if success else fail(msg))
            if body:
                print(f"   body: {body}")
    else:
        print(info("BACKEND_URL not set, skipping backend checks"))

    # httpbin (real IP through proxy)
    if "--geo" in sys.argv:
        success, msg, body = await check_http_via_proxy(
            proxy_url, "https://httpbin.org/ip"
        )
        print(ok(msg) if success else fail(msg))
        if body:
            print(f"   body: {body}")

    # httpbin general
    success, msg, body = await check_http_via_proxy(
        proxy_url, "https://httpbin.org/get?test=proxy_check"
    )
    print(ok(msg) if success else fail(msg))
    if body:
        print(f"   body: {body}")

    print()
    print(f"{BOLD}=== Done ==={RESET}")


if __name__ == "__main__":
    asyncio.run(main())
