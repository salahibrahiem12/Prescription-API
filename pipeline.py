"""Combine OCR and database suggestions without persisting prescription data."""
import re
from datetime import datetime, timezone
from prescription_agent import analyze_prescription
from drug_pipeline import build_index, match_many


def clean_drug_name(name: str) -> str:
    name = re.sub(r'\s*[\(\[].*', '', name)
    return re.split(r'\s*\d', name, maxsplit=1)[0].strip()


def build_meta(medicines, cleaned_names):
    return {name: {key: med.get(key) for key in
                  ('dose', 'frequency', 'duration', 'route', 'notes')}
            for med, name in zip(medicines, cleaned_names)}


def build_json_output(image_path, prescription, cleaned_names, medicines, meta, results):
    entries = []
    for med, clean in zip(medicines, cleaned_names):
        entries.append({
            'ocr_name': med.get('corrected_from') or med.get('name'),
            'cleaned_name': clean,
            **{key: med.get(key) for key in
               ('dose', 'frequency', 'duration', 'route', 'notes', 'read_confidence', 'corrected_from')},
            'matching': results.get(clean, {'matches': [], 'confidence': 'NOT_FOUND'}),
        })
    found = sum(bool(entry['matching'].get('matches')) for entry in entries)
    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'image': 'uploaded prescription',
        'ocr': {key: prescription.get(key) for key in
                ('doctor', 'patient', 'date', 'confidence', 'raw_text')},
        'medicines': entries,
        'summary': {'total': len(entries), 'found': found, 'not_found': len(entries) - found},
    }


def run(image_path: str, xlsx_path: str, k: int, json_only: bool = True):
    prescription = analyze_prescription(image_path)
    medicines = prescription.get('medicines') or []
    build_index(xlsx_path, force=False)
    entries = []
    # Match each line separately: a brand may appear at different doses.
    for med in medicines:
        clean = clean_drug_name(med.get('name') or '')
        meta = build_meta([med], [clean])
        matches = match_many([clean], k=k, meta=meta) if clean else {}
        entries.extend(build_json_output(image_path, prescription, [clean], [med], meta, matches)['medicines'])
    output = build_json_output(image_path, prescription, [], [], {}, {})
    output['medicines'] = entries
    found = sum(bool(entry['matching'].get('matches')) for entry in entries)
    output['summary'] = {'total': len(entries), 'found': found, 'not_found': len(entries) - found}
    return output
