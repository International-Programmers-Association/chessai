import sys, os, json

me = os.path.dirname(os.path.abspath(sys.argv[0] if getattr(sys, 'frozen', False) else __file__))

cfg_path = os.path.join(os.path.expanduser("~"), ".chessai", "config.json")
torch_dir = ""
if os.path.isfile(cfg_path):
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
        torch_dir = cfg.get("torch_dir", "")
    except Exception:
        pass

if not torch_dir:
    torch_dir = me

pkg = os.path.join(torch_dir, "torch")
if os.path.isdir(pkg) and os.path.isfile(os.path.join(pkg, "__init__.py")):
    if torch_dir not in sys.path:
        sys.path.insert(0, torch_dir)
elif os.path.isfile(os.path.join(torch_dir, "__init__.py")):
    parent = os.path.dirname(torch_dir)
    if parent not in sys.path:
        sys.path.insert(0, parent)

from chessai.app import main

if __name__ == "__main__":
    main()
