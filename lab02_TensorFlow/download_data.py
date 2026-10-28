"""Скачивает датасет THULab/SMAP (TsFile) с Hugging Face в data/raw/.
Запуск:  python download_data.py
"""
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(__file__).parent / "data" / "raw"
target.mkdir(parents=True, exist_ok=True)
path = snapshot_download(repo_id="THULab/SMAP", repo_type="dataset", local_dir=str(target))
print("Готово. Файлы в:", path)
for p in sorted(Path(path).rglob("*")):
    if p.is_file() and ".cache" not in p.parts:
        print(" ", p.relative_to(target), f"{p.stat().st_size/1024:.0f} КБ")
