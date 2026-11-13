"""Choose one fingerprinted face for each corpus.

Support is a cmap hit. A missing entry is reported and left out of every
timed sample. Faces are never mixed inside one corpus, and an empty
intersection is not a successful comparison.
"""

import json
import re
from pathlib import Path


FACE_ID = re.compile(r"[a-z][a-z0-9-]{0,40}")


class Face:
    def __init__(self, face_id, role, path, license_path, pitch, license_name):
        self.id = face_id
        self.role = role
        self.path = Path(path)
        self.license_path = Path(license_path)
        self.pitch = pitch
        self.license_name = license_name

    def record(self, digest):
        return {
            "id": self.id,
            "role": self.role,
            "name": self.path.name,
            "sha256": digest(self.path),
            "bytes": self.path.stat().st_size,
            "license_name": self.license_name,
            "license_sha256": digest(self.license_path),
            "license_bytes": self.license_path.stat().st_size,
            "pitch": self.pitch,
        }


def _assignment(text, label):
    face_id, separator, raw = text.partition("=")
    if not separator or not FACE_ID.fullmatch(face_id) or not raw:
        raise ValueError(label + " must be id=value with a lowercase id")
    return face_id, raw


def _pairs(items, label):
    found = {}
    for item in items:
        face_id, raw = _assignment(item, label)
        if face_id in found:
            raise ValueError("duplicate " + label + " for " + face_id)
        found[face_id] = raw
    return found


def _face(face_id, role, path, license_path, pitch, license_name):
    if pitch not in ("fixed", "cell"):
        raise ValueError(face_id + " pitch must be fixed or cell")
    if role not in ("primary", "coverage"):
        raise ValueError(face_id + " role must be primary or coverage")
    path = Path(path).expanduser().resolve()
    license_path = Path(license_path).expanduser().resolve()
    if not path.is_file() or not license_path.is_file():
        raise ValueError(face_id + " needs an existing font file and licence file")
    if not license_name:
        raise ValueError(face_id + " needs a licence name")
    return Face(face_id, role, path, license_path, pitch, license_name)


def resolve_faces(args):
    """The primary face is the default comparison. Coverage faces are opt-in."""
    if args.font_manifest:
        if args.font or args.font_license or args.coverage_font:
            raise ValueError("a font manifest replaces --font and --coverage-font")
        payload = json.loads(Path(args.font_manifest).expanduser().read_text(encoding="utf-8"))
        rows = payload.get("faces") if isinstance(payload, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("font manifest needs a faces list")
        faces = []
        for row in rows:
            faces.append(_face(
                row["id"], row["role"], row["file"], row["license"],
                row.get("pitch", "fixed"), row.get("license_name", "")))
    else:
        if not args.font or not args.font_license:
            raise ValueError("font suite needs --font and --font-license, or a manifest")
        faces = [_face("primary", "primary", args.font, args.font_license, "fixed",
                       args.font_license_name or "recorded licence file")]
        licenses = _pairs(args.coverage_license, "coverage licence")
        pitches = _pairs(args.coverage_pitch, "coverage pitch")
        names = _pairs(args.coverage_license_name, "coverage licence name")
        for item in args.coverage_font:
            face_id, raw = _assignment(item, "coverage font")
            if face_id == "primary" or any(face.id == face_id for face in faces):
                raise ValueError("duplicate coverage face " + face_id)
            if face_id not in licenses or face_id not in names:
                raise ValueError(face_id + " needs a licence file and a licence name")
            faces.append(_face(face_id, "coverage", raw, licenses[face_id],
                               pitches.get(face_id, "fixed"), names[face_id]))
    primaries = [face for face in faces if face.role == "primary"]
    if len(primaries) != 1 or faces[0].role != "primary":
        raise ValueError("the face list needs exactly one primary face, listed first")
    if len({face.id for face in faces}) != len(faces):
        raise ValueError("duplicate face id")
    return faces


def _split(codes, supported):
    if supported is None:
        return [], list(codes)
    if len(supported) != len(codes):
        raise ValueError("coverage probe length mismatch")
    present = [code for code, yes in zip(codes, supported) if yes]
    missing = [code for code, yes in zip(codes, supported) if not yes]
    return present, missing


def coverage_plan(corpora, unsupported, faces, probes):
    """Return a report and the corpora that may be timed.

    ``probes[face_id][corpus]`` is the cmap boolean list. The primary face
    wins whenever it has any shared scalar. A coverage face is used only
    when the primary intersection is empty, and only for its own intersection.
    """
    report, jobs = {}, []
    for name, codes in corpora.items():
        matrix = []
        for order, face in enumerate(faces):
            present, missing = _split(codes, probes.get(face.id, {}).get(name))
            matrix.append({
                "id": face.id,
                "role": face.role,
                "pitch": face.pitch,
                "order": order,
                "present": len(present),
                "missing": len(missing),
                "missing_scalars": missing,
                "present_scalars": present,
            })
        entry = {
            "requested": list(codes),
            "requested_count": len(codes),
            "faces": [{key: value for key, value in row.items() if key != "present_scalars"}
                      for row in matrix],
            "status": "unavailable",
            "action": "not-timed",
            "shared": [],
            "freetype_missing": matrix[0]["missing_scalars"] if matrix else list(codes),
            "comparison_face": None,
        }
        if name in unsupported or not codes:
            entry["reason"] = "not declared for selected weights"
            entry["note"] = "Weights outside the reviewed profile are not timed."
        elif matrix[0]["present"]:
            entry.update(status="measured", action="timed-against-primary",
                         comparison_face=matrix[0]["id"], shared=matrix[0]["present_scalars"],
                         freetype_missing=matrix[0]["missing_scalars"],
                         note="Timed scalars are the primary face's cmap intersection. Missing entries were not drawn.")
        else:
            choices = [row for row in matrix[1:] if row["present"]]
            if not choices:
                entry["reason"] = "no declared face contains this corpus"
                entry["note"] = ("No cmap entry is available in the declared faces. "
                                 "Fallback glyphs were not drawn or timed.")
            else:
                chosen = max(choices, key=lambda row: (row["present"], -row["order"]))
                entry.update(status="measured", action="timed-against-coverage-face",
                             comparison_face=chosen["id"], shared=list(chosen["present_scalars"]),
                             freetype_missing=list(chosen["missing_scalars"]),
                             note=("The primary face has no cmap entries here. "
                                   "The comparison uses the named coverage face only. "
                                   "Missing entries were not drawn or timed."))
        if entry["status"] == "measured":
            jobs.append((name, entry["comparison_face"], list(entry["shared"])))
        report[name] = entry
    return report, jobs
