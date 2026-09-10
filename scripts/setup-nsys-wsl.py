"""Apply NVIDIA's documented Nsight Systems timestamp workaround on WSL.

https://docs.nvidia.com/nsight-systems/2025.3/ReleaseNotes/index.html
Only the CuptiUseRawGpuTimestamps option is changed; other settings are preserved.
"""
from pathlib import Path
import platform
import re
import shutil
import subprocess

if "microsoft" not in platform.release().lower():
    raise SystemExit("This workaround is only intended for WSL.")
executable = shutil.which("nsys")
if not executable:
    raise SystemExit("nsys is not on PATH; source scripts/oxide-env.sh first.")
config = Path(subprocess.check_output([executable, "-z"], text=True).strip())
config.parent.mkdir(parents=True, exist_ok=True)
content = config.read_text() if config.exists() else ""
setting = "CuptiUseRawGpuTimestamps=false"
if re.search(r"(?m)^\s*CuptiUseRawGpuTimestamps\s*=", content):
    content = re.sub(r"(?m)^\s*CuptiUseRawGpuTimestamps\s*=.*$", setting, content)
else:
    content = content.rstrip() + "\n" + setting + "\n"
config.write_text(content)
print(f"Updated {config}: {setting}")
