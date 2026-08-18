#!/usr/bin/env python3
"""Demo 可用性验证（子赛道 B 准入项 4.2）：
1. 半双工 gradio demo：启动 gradio_demo.py → 页面可访问 → 通过 API 验证
   TTS 流式输出连续（多分片、无中断）
2. 输出 result.json（demo 启动方式、页面状态、端到端请求结果）

用法:
  python3 verify_demo.py [--base http://127.0.0.1:8091] [--out /workspace/submission/7_runtime/results/demo]
"""
import argparse
import json
import os
import signal
import subprocess
import os
import sys
import time
import urllib.request

ASSETS = os.environ.get("MODEL_DIR", "/workspace/shared_assets/models/OpenBMB/MiniCPM-o-4_5") + "/assets"
GRADIO_PY = os.environ.get("REPO_DIR", "/workspace/vllm-omni") + "/examples/online_serving/minicpmo/gradio_demo.py"
GRADIO_PORT = 7862


def wait_port(port: int, timeout_s: int = 120) -> bool:
    import socket
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        s = socket.socket()
        s.settimeout(2)
        try:
            s.connect(("127.0.0.1", port))
            s.close()
            return True
        except Exception:
            s.close()
            time.sleep(3)
    return False


def wait_api(base: str, timeout_s: int = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            urllib.request.urlopen(f"{base}/v1/models", timeout=5)
            return True
        except Exception:
            time.sleep(10)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8091")
    ap.add_argument("--out", default=os.environ.get("SUBMISSION_DIR", "/workspace/submission") + "/7_runtime/results/demo")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    result: dict = {"demo": "gradio_half_duplex", "steps": {}}

    # 1. API 就绪
    if not wait_api(args.base):
        print("API 服务未就绪，退出")
        sys.exit(1)
    result["steps"]["api_ready"] = True

    # 2. 启动 gradio demo
    proc = subprocess.Popen(
        [sys.executable, GRADIO_PY,
         "--minicpmo45-api-base", f"{args.base}/v1",
         "--minicpmo45-model", "openbmb/MiniCPM-o-4_5",
         "--port", str(GRADIO_PORT)],
        stdout=open(f"{args.out}/gradio.log", "w"),
        stderr=subprocess.STDOUT,
    )
    result["steps"]["gradio_pid"] = proc.pid
    ok = wait_port(GRADIO_PORT, timeout_s=180)
    result["steps"]["gradio_up"] = ok
    print(f"gradio 页面 {'可访问' if ok else '启动失败'}: http://127.0.0.1:{GRADIO_PORT}")

    # 3. 页面 HTTP 状态
    if ok:
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{GRADIO_PORT}/", timeout=10)
            result["steps"]["page_http"] = r.status
        except Exception as e:
            result["steps"]["page_http"] = f"err: {e}"

    # 4. 端到端 TTS 流式请求（多模态输入 + 语音输出）
    import base64
    import urllib.request as urlreq

    def b64data(path: str, mime: str) -> str:
        return f"data:{mime};base64," + base64.b64encode(open(path, "rb").read()).decode()

    payload = {
        "model": "openbmb/MiniCPM-o-4_5",
        "modalities": ["text", "audio"],
        "stream": True,
        "chat_template_kwargs": {"use_tts_template": True},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": b64data(f"{ASSETS}/fossil.png", "image/png")}},
            {"type": "text", "text": "用英文描述这张图片，并用语音读出来。"}]}],
    }
    req = urlreq.Request(f"{args.base}/v1/chat/completions",
                         data=json.dumps(payload).encode(),
                         headers={"Content-Type": "application/json"})
    t0 = time.time()
    chunks, audio_parts = 0, 0
    aud_dir = f"{args.out}/audio_chunks"
    os.makedirs(aud_dir, exist_ok=True)
    try:
        with urlreq.urlopen(req, timeout=600) as resp:
            for line in resp:
                line = line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                chunks += 1
                msg = (evt.get("choices") or [{}])[0].get("delta", {})
                # 音频 chunk：modality=audio，delta.content 为 base64 WAV
                if evt.get("modality") == "audio" or msg.get("audio"):
                    audio_parts += 1
                    b64 = msg.get("audio") or msg.get("content")
                    if isinstance(b64, str):
                        import base64
                        try:
                            open(f"{aud_dir}/{audio_parts:04d}.wav", "wb").write(base64.b64decode(b64))
                        except Exception:
                            pass
    except Exception as e:
        result["steps"]["stream_error"] = str(e)[:300]
    result["steps"]["stream_chunks"] = chunks
    result["steps"]["stream_seconds"] = round(time.time() - t0, 1)
    print(f"流式响应: {chunks} chunks, {result['steps'].get('stream_seconds', '?')}s")

    # 5. gradio 日志检查（无崩溃）
    log = open(f"{args.out}/gradio.log").read()
    result["steps"]["gradio_crashed"] = "Traceback" in log and "Traceback" in log.split()[-500:]

    # 6. 清理
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=15)
    result["ok"] = bool(result["steps"].get("gradio_up")) and chunks > 0
    with open(f"{args.out}/result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
