import os
import sys
import subprocess
from glob import glob

def setup_java_notebook():
    """
    Automatically configure Java for PyJNIus / Ashlar in a conda environment.
    Detects libjvm.so inside the environment and sets JAVA_HOME, PATH, and JVM_PATH.
    """
    env_prefix = os.path.dirname(os.path.dirname(sys.executable))

    # Search for the JDK folder inside conda env
    jdk_folders = glob(os.path.join(env_prefix, "lib/jvm/*"))
    if not jdk_folders:
        raise RuntimeError("No JDK found in the environment. Ensure openjdk is installed.")

    jdk_home = jdk_folders[0]  # take first match
    os.environ["JAVA_HOME"] = jdk_home
    os.environ["PATH"] = os.path.join(jdk_home, "bin") + ":" + os.environ["PATH"]

    # Locate libjvm.so
    jvm_candidates = glob(os.path.join(jdk_home, "lib", "server", "libjvm.so"))
    if jvm_candidates:
        os.environ["JVM_PATH"] = jvm_candidates[0]
        print("JVM_PATH set to:", os.environ["JVM_PATH"])
    else:
        raise RuntimeError(
            "libjvm.so not found inside the JDK. You may need to install openjdk properly."
        )

    # Verify javac
    try:
        result = subprocess.run(["javac", "-version"], capture_output=True, text=True, check=True)
        javac_version = result.stderr.strip() if result.stderr else result.stdout.strip()
    except FileNotFoundError:
        javac_version = "javac not found in PATH!"

    print("JAVA_HOME set to:", os.environ["JAVA_HOME"])
    print("javac version:", javac_version)
    print("Java setup complete for this notebook session!")
