# Dependency files

`constraints-py310.txt` pins the project's direct inference, training, and
test dependencies. Use it with `pip install -c ... -e ".[train,test]"`.
It is a constraints file, not a standalone list of packages to install.

`linux-cu121-py310.lock.txt` records the complete resolved reference
environment, including transitive packages. It targets Linux x86_64,
Python 3.10, and CUDA 12.1 wheels; do not use it for the CPU-only or
Windows installation paths. The editable project itself is intentionally
excluded so the lock contains no local source path.

See `docs/ENVIRONMENT.md` for installation and validation commands.
