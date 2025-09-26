import os
import subprocess
import shutil

def setup_java_env(manual_java_home="/usr/lib/jvm/java-11-openjdk-amd64"):
    """
    Ensure JAVA_HOME and JVM_PATH are set.

    Order of preference:
    1. If java is already callable, do nothing.
    2. Otherwise, try to detect JAVA_HOME from system java.
    3. If that fails, fall back to manual_java_home.

    Returns
    -------
    dict
        Dictionary with JAVA_HOME and JVM_PATH.
    """

    # 1. If java already works → do nothing
    try:
        subprocess.run(["java", "-version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("Java is already available in this environment (no changes made).")
        return {
            "JAVA_HOME": os.environ.get("JAVA_HOME"),
            "JVM_PATH": os.environ.get("JVM_PATH")
        }
    except Exception:
        print("Java not available, trying to configure...")

    # 2. Try to auto-detect JAVA_HOME from java path
    java_exe = shutil.which("java")
    if java_exe:
        java_home = os.path.dirname(os.path.dirname(os.path.realpath(java_exe)))
        print(f"Auto-detected JAVA_HOME: {java_home}")
    else:
        # 3. Fallback to manual path
        java_home = manual_java_home.rstrip("/")
        print(f"Falling back to manual JAVA_HOME: {java_home}")

    jvm_path = os.path.join(java_home, "lib", "server", "libjvm.so")

    # Set environment variables
    os.environ["JAVA_HOME"] = java_home
    os.environ["JVM_PATH"] = jvm_path
    os.environ["PATH"] = os.path.join(java_home, "bin") + ":" + os.environ["PATH"]

    print(f"JAVA_HOME set to: {java_home}")
    print(f"JVM_PATH set to: {jvm_path}")

    # Confirm setup
    try:
        subprocess.run(["java", "-version"], check=True)
    except Exception:
        print("Warning: java still not available after setup")

    print("Java environment:", java_home, jvm_path)

    return {"JAVA_HOME": java_home, "JVM_PATH": jvm_path}
