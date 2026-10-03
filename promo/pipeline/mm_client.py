"""Thin MiniMax API client (CN region, Token Plan subscription key).

The key is read from the MINIMAX_API_KEY environment variable, or from the
file named by MINIMAX_API_KEY_FILE. It is never written into the repo.
"""
import json
import os
import time

import requests

BASE = "https://api.minimax.cn"


def _key():
    k = os.environ.get("MINIMAX_API_KEY")
    if not k and os.environ.get("MINIMAX_API_KEY_FILE"):
        with open(os.environ["MINIMAX_API_KEY_FILE"]) as f:
            k = f.read().strip()
    if not k:
        raise SystemExit("set MINIMAX_API_KEY or MINIMAX_API_KEY_FILE")
    return k


def _post(path, payload, timeout=600, retries=4, headers=None):
    h = {"Authorization": f"Bearer {_key()}", "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    last = None
    for attempt in range(retries):
        try:
            r = requests.post(BASE + path, headers=h, json=payload, timeout=timeout)
            data = r.json()
        except (requests.RequestException, ValueError) as e:
            last = e
            time.sleep(2 ** (attempt + 1))
            continue
        code = (data.get("base_resp") or {}).get("status_code", 0)
        if code in (1002, 1039):  # rate limited
            last = data
            time.sleep(10 * (attempt + 1))
            continue
        return data
    raise RuntimeError(f"MiniMax request failed: {last}")


def chat(messages, system=None, model="MiniMax-M3.1-Flash-Preview", max_tokens=64000, temperature=None,
         effort="max"):
    """Anthropic-compatible Messages API. Returns (text, thinking).

    M3.1-Flash-Preview always thinks; thinking tokens count against max_tokens.
    """
    payload = {"model": model, "max_tokens": max_tokens, "messages": messages}
    if effort and "M3.1" in model:
        payload["output_config"] = {"effort": effort}
    if system:
        payload["system"] = system
    if temperature is not None:
        payload["temperature"] = temperature
    data = _post("/anthropic/v1/messages", payload, timeout=1800,
                 headers={"x-api-key": _key(), "anthropic-version": "2023-06-01"})
    if data.get("type") == "error":
        raise RuntimeError(data)
    text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
    thinking = "".join(b.get("thinking", "") for b in data["content"] if b.get("type") == "thinking")
    return text, thinking


def image(prompt, out_paths, width=2048, height=864, seed=None, model="image-01"):
    """Text-to-image. Writes len(out_paths) images; returns the list written."""
    payload = {"model": model, "prompt": prompt, "width": width, "height": height,
               "n": len(out_paths), "response_format": "url", "prompt_optimizer": False}
    if seed is not None:
        payload["seed"] = seed
    data = _post("/v1/image_generation", payload)
    if data.get("base_resp", {}).get("status_code") != 0:
        raise RuntimeError(data)
    urls = data["data"]["image_urls"]
    written = []
    for url, p in zip(urls, out_paths):
        r = requests.get(url, timeout=300)
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)
        written.append(p)
    return written


def tts(text, out_path, voice_id, model="speech-2.8-hd", speed=1.0, vol=1.0, pitch=0,
        emotion=None, voice_modify=None, sample_rate=44100, fmt="wav"):
    """Synchronous T2A v2. Writes audio file, returns extra_info."""
    vs = {"voice_id": voice_id, "speed": speed, "vol": vol, "pitch": pitch}
    if emotion:
        vs["emotion"] = emotion
    payload = {"model": model, "text": text, "stream": False, "voice_setting": vs,
               "audio_setting": {"sample_rate": sample_rate, "format": fmt, "channel": 1},
               "output_format": "hex", "language_boost": "auto"}
    if voice_modify:
        payload["voice_modify"] = voice_modify
    data = _post("/v1/t2a_v2", payload)
    if data.get("base_resp", {}).get("status_code") != 0:
        raise RuntimeError(data)
    with open(out_path, "wb") as f:
        f.write(bytes.fromhex(data["data"]["audio"]))
    return data.get("extra_info", {})


if __name__ == "__main__":
    print(json.dumps(chat([{"role": "user", "content": "ping"}], max_tokens=50)[0]))
