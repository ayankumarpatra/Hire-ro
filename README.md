# HireLens AI - Final Local Build

HireLens AI is a small Flask hackathon application for screening multiple PDF resumes against a job description.

## Final features

- Multi-PDF upload with server-side validation
- Local PDF text extraction with PyMuPDF
- Local resume information extraction
- Local explainable Profile Fit percentage
- Candidate ranking
- Search candidates
- Filter by score band
- Sort by score or name
- Mandatory-skill flags
- Expandable candidate details
- Loading/checking screen while analysis is running
- No Gemini/API call in this build
- No database; uploaded resume content is processed in memory

## Score

The local score is a heuristic:

- 60% job-description text similarity (TF-IDF + cosine similarity)
- 30% detected skill coverage
- 10% resume evidence sections

The percentage is **not** a probability of hiring success and does not verify candidate claims.

## Run on Windows PowerShell

```powershell
py -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:FLASK_DEBUG="0"
python app.py
```

Open `http://127.0.0.1:5000`.
