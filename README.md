# pyBgRemover script - mac
This set of installation instructions and comands following are exclusively for installing and using this script on mac osx

# Install dependencies

```
pip install -r requirements.txt
```

# Run

No model download needed - InSPyReNet weights are fetched automatically on first run.

```
python pyBgRemover.py input.jpg -o output.jpg
```

Options: `--outline 14 --outlinecolor "#fff" --blur 5 --darken 0.55 --fade 0.4 --mode base`
