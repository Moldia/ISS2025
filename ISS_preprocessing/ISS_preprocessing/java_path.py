import os
import shutil
import subprocess
from pathlib import Path
from glob import glob
import re

def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT)
    except Exception:
        return ""

def _java_version_from_home(home: Path):
    java = home / "bin" / "java"
    if not java.exists():
        return (-1, "")  # unknown
    out = _run([str(java), "-version"])
    # Example first line: openjdk version "17.0.10"  or  java version "1.8.0_392"
    m = re.search(r'version\s+"([\d._]+)"', out)
    ver = m.group(1) if m else ""
    # Normalize major (e.g., 17 from 17.0.10, 8 from 1.8.0_392)
    try:
        major = int(ver.split(".")[0]) if not ver.startswith("1.") else int(ver.split(".")[1])
    except Exception:
        major = -1
    return (major, ver)

def _has_javac(home: Path):
    return (home / "bin" / "javac").exists()

def _find_libjvm(home: Path):
    # Common Linux layouts
    candidates = [
        home / "lib" / "server" / "libjvm.so",
        home / "jre" / "lib" / "server" / "libjvm.so",
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Fallback: recursive search within home (bounded)
    hits = glob(str(home / "**" / "libjvm.so"), recursive=True)
    return Path(hits[0]).resolve() if hits else None

def _possible_java_homes():
    homes = []

    # 1) Respect current env if present
    if os.environ.get("JAVA_HOME"):
        homes.append(Path(os.environ["JAVA_HOME"]))

    # 2) From executables on PATH
    for exe in ("javac", "java"):
        p = shutil.which(exe)
        if p:
            homes.append(Path(p).resolve().parent.parent)

    # 3) Debian/Ubuntu alternatives
    for alt in ("javac", "java"):
        out = _run(["update-alternatives", "--list", alt])
        for line in out.splitlines():
            # e.g., /usr/lib/jvm/java-17-openjdk-amd64/bin/javac
            path = Path(line.strip()).resolve()
            if path.name in ("java", "javac"):
                homes.append(path.parent.parent)

    # 4) Common roots to scan one level deep
    roots = [
        "/usr/lib/jvm",
        "/usr/java",
        "/opt/java",
        os.environ.get("CONDA_PREFIX", ""),  # conda openjdk
    ]
    for r in filter(None, roots):
        rp = Path(r)
        if rp.is_dir():
            # add subdirectories that look like Java installations
            for sub in rp.iterdir():
                if sub.is_dir():
                    homes.append(sub)

    # Dedup while preserving order
    seen, uniq = set(), []
    for h in homes:
        try:
            h = h.resolve()
        except Exception:
            continue
        if h not in seen:
            seen.add(h)
            uniq.append(h)
    return uniq

def _score_home(home: Path):
    """
    Higher score = more preferred.
    Prefer JDK (has javac), then higher major version (21 > 17 > 11 > 8), then others.
    """
    major, _ = _java_version_from_home(home)
    score = 0
    if _has_javac(home):
        score += 1000
    # weight major version
    pref = {21: 300, 17: 200, 11: 150, 8: 100}
    score += pref.get(major, major if major > 0 else 0)
    # small bonus if path name looks like a proper OpenJDK/JDK
    s = home.as_posix().lower()
    for kw in ("openjdk", "jdk", "temurin", "zulu", "liberica", "graal"):
        if kw in s:
            score += 5
    return score

def _java_runs(home: Path, timeout: int = 5) -> bool:
    """
    Returns True if <home>/bin/java -version executes successfully.
    """
    java = home / "bin" / "java"
    if not java.exists():
        return False
    try:
        subprocess.run([str(java), "-version"],
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,
                       check=True,
                       timeout=timeout)
        return True
    except Exception:
        return False


def detect_java(set_env: bool = True,
                min_major: int | None = None,
                require_jdk: bool = False):
    """
    Find and verify a usable Java installation.
    Automatically sets JAVA_HOME, JVM_PATH, and PATH if requested.
    Prints detected configuration details (no return value).
    """
    candidates = []
    for h in _possible_java_homes():
        if not (h / "bin" / "java").exists():
            continue

        if require_jdk and not _has_javac(h):
            continue
        if min_major is not None:
            major, _ = _java_version_from_home(h)
            if major < 0 or major < min_major:
                continue

        jvm = _find_libjvm(h)
        if not jvm:
            continue
        candidates.append((h, jvm))

    if not candidates:
        raise RuntimeError("No Java installations with a detectable libjvm were found.")

    ranked = sorted(candidates, key=lambda t: _score_home(t[0]), reverse=True)

    for home, jvm in ranked:
        if _java_runs(home):
            if set_env:
                os.environ["JAVA_HOME"] = str(home)
                os.environ["JVM_PATH"]  = str(jvm)
                os.environ["PATH"]      = f"{home / 'bin'}:{os.environ.get('PATH','')}"

            print("Java environment detected:")
            print(f"   JAVA_HOME: {home}")
            print(f"   JVM_PATH : {jvm}\n")
            return  # No return value

    raise RuntimeError("No runnable Java installation found (all candidates failed).")
