# Project Run Instructions

Follow these steps from the project root in Windows PowerShell.

1. Create the virtual environment:

```powershell
python -m venv .venv
```

2. Activate the virtual environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

3. Install the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

4. Start the backend and frontend together:

```powershell
powershell -ExecutionPolicy ByPass -File .\run.ps1
```

5. Open the app in your browser:

```text
http://127.0.0.1:8501
```

Backend API URL:

```text
http://127.0.0.1:8000
```

## Notes

- If `python` points to the wrong interpreter, use the full venv path shown above.
- If the venv already exists, skip step 1.
- If you change dependencies, rerun step 3.
- If `No module named pip` appears, run `.\.venv\Scripts\python.exe -m ensurepip --upgrade` and then rerun step 3.
