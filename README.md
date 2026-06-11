## Установка

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте его:

```bash
# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate
```
Для запуска на GPU нужно установить torch с соответствующей версией cuda. Для CPU этот шаг пропустите

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Установите зависимости:
```bash
pip install -r requirements.txt
```
