import os

ROOT = "."

for root, dirs, files in os.walk(ROOT):
    for file in files:
        if file.endswith((".py", ".md", ".yaml", ".yml", ".txt")):
            path = os.path.join(root, file)
            try:
                with open(path, "rb") as f:
                    content = f.read()

                # decode ignoring bad chars, then re-save clean UTF-8
                cleaned = content.decode("utf-8", errors="ignore")

                with open(path, "w", encoding="utf-8") as f:
                    f.write(cleaned)

                print(f"Fixed: {path}")

            except Exception as e:
                print(f"Skipped: {path} -> {e}")