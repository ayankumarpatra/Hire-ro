"""HireLens AI - Application Controller."""

import os
import re

from flask import Flask, render_template, request

from validators import (
    load_limits,
    clean_filename,
    validate_extension,
    read_upload_bytes,
    validate_pdf_bytes,
    select_uploaded_files,
    UploadError,
)
from resume_parser import parse_resume_pdf, ResumeParseError
from matching import calculate_match

app = Flask(__name__)
limits = load_limits()

MAX_REQUEST_BYTES = (limits.max_files * (limits.max_file_bytes // (1024 * 1024) + 1) + 1) * 1024 * 1024
app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES

MAX_JOB_DESCRIPTION_CHARS = 10_000
MAX_SKILLS_CHARS = 500
MAX_SKILLS = 20
MAX_SKILL_LENGTH = 50


def page_context(**extra):
    context = {
        "max_files": limits.max_files,
        "max_file_mb": limits.max_file_bytes // (1024 * 1024),
        "max_pages": limits.max_pages,
        "max_jd_chars": MAX_JOB_DESCRIPTION_CHARS,
        "max_skills_chars": MAX_SKILLS_CHARS,
        "errors": [],
        "job_description": "",
        "mandatory_skills": "",
    }
    context.update(extra)
    return context


def parse_skill_list(raw_text):
    skills = []
    seen = set()
    for part in re.split(r"[,;\n]", raw_text):
        skill = part.strip()
        if skill and skill.casefold() not in seen:
            seen.add(skill.casefold())
            skills.append(skill)
    return skills


def error_page(message, status_code):
    return render_template("index.html", **page_context(errors=[message])), status_code


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", **page_context())


@app.route("/analyze", methods=["POST"])
def analyze():
    job_description = request.form.get("job_description", "").strip()
    mandatory_skills_raw = request.form.get("mandatory_skills", "").strip()
    raw_files = request.files.getlist("resumes")

    errors = []
    if not job_description:
        errors.append("Paste the job description.")
    elif len(job_description) > MAX_JOB_DESCRIPTION_CHARS:
        errors.append(f"Job description exceeds {MAX_JOB_DESCRIPTION_CHARS:,} characters.")

    skills = []
    if len(mandatory_skills_raw) > MAX_SKILLS_CHARS:
        errors.append("Mandatory skills field is too long.")
    else:
        skills = parse_skill_list(mandatory_skills_raw)
        if len(skills) > MAX_SKILLS:
            errors.append(f"List at most {MAX_SKILLS} mandatory skills.")
        elif any(len(s) > MAX_SKILL_LENGTH for s in skills):
            errors.append(f"Each skill must be under {MAX_SKILL_LENGTH} characters.")

    try:
        chosen_files = select_uploaded_files(raw_files, limits)
    except UploadError as e:
        errors.append(e.message)
        chosen_files = []

    if errors:
        return render_template(
            "index.html",
            **page_context(
                errors=errors,
                job_description=job_description,
                mandatory_skills=mandatory_skills_raw,
            ),
        ), 400

    results = []
    readable_count = 0

    for file_storage in chosen_files:
        safe_name = clean_filename(file_storage.filename)
        file_result = {
            "filename": safe_name,
            "success": False,
            "error_message": "",
            "page_count": 0,
            "char_count": 0,
            "assessment_message": None,
            "info": None,
            "match": None,
        }

        try:
            validate_extension(file_storage.filename)
            file_bytes = read_upload_bytes(file_storage, limits.max_file_bytes)
            validate_pdf_bytes(file_bytes, limits)
            parsed_resume = parse_resume_pdf(file_bytes, limits)

            file_result["success"] = True
            file_result["page_count"] = parsed_resume.page_count
            file_result["char_count"] = parsed_resume.char_count
            file_result["assessment_message"] = parsed_resume.assessment.message
            file_result["info"] = parsed_resume.info
            file_result["match"] = calculate_match(
                job_description,
                skills,
                parsed_resume.text,
                parsed_resume.info,
            )
            readable_count += 1

        except (UploadError, ResumeParseError) as e:
            file_result["error_message"] = e.message
        except Exception:
            file_result["error_message"] = "An unexpected error occurred while processing this file."

        results.append(file_result)

    results.sort(key=lambda item: (item.get("match") or {}).get("score", -1), reverse=True)

    return render_template(
        "results.html",
        results=results,
        readable=readable_count,
        total=len(chosen_files),
    )


@app.errorhandler(404)
def not_found(error):
    return error_page("That page does not exist.", 404)


@app.errorhandler(405)
def method_not_allowed(error):
    return error_page("Invalid request method.", 405)


@app.errorhandler(413)
def too_large(error):
    return error_page("The upload exceeds the size limit.", 413)


@app.errorhandler(500)
def server_error(error):
    return error_page("Something went wrong on our side. Try again in a moment.", 500)


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; frame-ancestors 'none'; form-action 'self'"
    return response


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG") == "1"
    app.run(host="127.0.0.1", port=5000, debug=debug_mode)
