import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import drug_pipeline as drugs
import gemini_runtime
import pipeline
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image
import main


class MatchingRegressionTests(unittest.TestCase):
    def test_brand_cleaning(self):
        for original, expected in [('Nexium 40 mg', 'Nexium'), ('Lasix', 'Lasix'),
                                   ('Lasix 2.5 mg', 'Lasix'), ('Gliben5', 'Gliben')]:
            with self.subTest(original=original):
                self.assertEqual(pipeline.clean_drug_name(original), expected)

    def test_decimal_strength_before_limit_exact_fuzzy_and_ai(self):
        products = ['TESTBRAND 25 MG TABLETS', 'TESTBRAND 2.5 MG TABLETS']
        index = {'brand_map': {'TESTBRAND': products}, 'names': products}
        ai = {'input_type': 'brand', 'active_ingredient': '', 'category': 'rx',
              'brands': ['TESTBRAND'], 'confidence': 'high'}
        with patch.object(drugs, '_load', return_value=index), patch.object(drugs, '_gemini_lookup', return_value=ai):
            for query in ['TESTBRAND', 'TESTBRAN', 'unrecognizable']:
                for k in [1, 5]:
                    with self.subTest(query=query, k=k):
                        result = drugs.match(query, k=k, dose='2.5 mg')
                        self.assertEqual(result['matches'][0], products[1])
                        self.assertLessEqual(len(result['matches']), k)
            self.assertEqual(drugs.match('TESTBRAND 2.5 mg', k=1)['matches'], [products[1]])

    def test_pack_count_is_not_strength(self):
        candidates = ['TESTBRAND 25 MG 2.5 TABLETS', 'TESTBRAND 2.5 MG 25 TABLETS']
        self.assertEqual(drugs._rerank(candidates, 'oral', '2.5 mg')[0], candidates[1])
        unitless = ['TESTBRAND 25 TABLETS', 'TESTBRAND 2.5 TABLETS']
        self.assertEqual(drugs._rerank(unitless, 'oral', '2.5 mg')[0], unitless[1])

    def test_uncertain_and_repeated_lines_preserved_without_disk_or_logs(self):
        prescription = {'patient': 'FICTIONAL TEST ONLY', 'medicines': [
            {'name': 'Nexium 40 mg', 'dose': '40 mg'},
            {'name': 'Nexium 20 mg', 'dose': '20 mg'},
            {'name': None, 'read_confidence': 'low', 'notes': 'Unreadable fictional line'}]}
        def matches(names, k, meta):
            return {names[0]: {'matches': [meta[names[0]]['dose']], 'confidence': 'LOW'}}
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(out), patch.object(pipeline, 'analyze_prescription', return_value=prescription), patch.object(pipeline, 'build_index'), patch.object(pipeline, 'match_many', side_effect=matches):
            result = pipeline.run(str(Path(folder) / 'fictional.png'), main.XLSX_PATH, 5)
            self.assertEqual(list(Path(folder).iterdir()), [])
        self.assertEqual(out.getvalue(), '')
        self.assertEqual(len(result['medicines']), 3)
        self.assertEqual(result['medicines'][0]['matching']['matches'], ['40 mg'])
        self.assertEqual(result['medicines'][1]['matching']['matches'], ['20 mg'])
        self.assertEqual(result['medicines'][2]['matching']['confidence'], 'NOT_FOUND')


class WebsiteApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client_context = TestClient(main.app)
        cls.client = cls.client_context.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)

    def setUp(self):
        main._client_calls.clear()
        gemini_runtime._calls.clear()

    def image(self):
        buffer = io.BytesIO()
        Image.new('RGB', (640, 480), 'white').save(buffer, format='PNG')
        return buffer.getvalue()

    def test_health_website_and_swagger(self):
        self.assertIs(self.client.get('/health').json()['index_ready'], True)
        response = self.client.get('/')
        self.assertIn('Google Gemini processes', response.text)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertEqual(self.client.get('/static/app.js').status_code, 200)
        self.assertEqual(self.client.get('/docs').status_code, 200)
        self.assertEqual(self.client.get('/openapi.json').status_code, 200)

    def test_real_panadol_lookup_without_gemini(self):
        with patch.object(drugs, '_gemini_lookup', side_effect=AssertionError('Panadol must match locally')):
            response = self.client.post('/match', json={'names': ['Panadol'], 'k': 5})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()['results']['Panadol']['matches'])

    def test_input_bounds(self):
        for body in [{'names': []}, {'names': [' ']}, {'names': ['x'], 'k': 0},
                     {'names': ['x'], 'k': 21}, {'names': ['x'] * 11}]:
            self.assertEqual(self.client.post('/match', json=body).status_code, 422)

    def test_invalid_and_oversized_uploads(self):
        self.assertEqual(self.client.post('/analyze', files={'file': ('fake.png', b'not an image', 'image/png')}).status_code, 422)
        self.assertEqual(self.client.post('/analyze', files={'file': ('file.txt', b'x', 'text/plain')}).status_code, 415)
        self.assertEqual(self.client.post('/analyze', files={'file': ('big.png', b'x' * (main.MAX_UPLOAD + 1), 'image/png')}).status_code, 413)
        self.assertEqual(self.client.post('/analyze', content=b'x', headers={'Content-Length': str(main.MAX_UPLOAD + 65537)}).status_code, 413)

    def test_missing_key_is_explicit(self):
        with patch.dict(os.environ, {'GEMINI_API_KEY': ''}):
            response = self.client.post('/analyze', files={'file': ('fictional.png', self.image(), 'image/png')})
        self.assertEqual(response.status_code, 503)
        self.assertIn('Live OCR is pending', response.json()['detail'])

    def test_upload_success_and_failure_cleanup(self):
        paths = []
        def fake_run(path, *args):
            paths.append(Path(path))
            self.assertTrue(Path(path).exists())
            return {'medicines': [{'ocr_name': 'Panadol', 'dose': '500 mg'}]}
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'fictional-test-value'}), patch.object(main, 'run', side_effect=fake_run):
            response = self.client.post('/analyze', files={'file': ('fictional.png', self.image(), 'image/png')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['medicines'][0]['ocr_name'], 'Panadol')
        self.assertFalse(paths[0].exists())
        def fail_run(path, *args):
            paths.append(Path(path))
            raise RuntimeError('FICTIONAL PRIVATE DETAILS')
        with patch.dict(os.environ, {'GEMINI_API_KEY': 'fictional-test-value'}), patch.object(main, 'run', side_effect=fail_run):
            response = self.client.post('/analyze', files={'file': ('fictional.png', self.image(), 'image/png')})
        self.assertEqual(response.status_code, 502)
        self.assertNotIn('PRIVATE', response.text)
        self.assertFalse(paths[-1].exists())

    def test_rate_limits(self):
        with patch.object(drugs, 'match_many', return_value={}):
            for _ in range(5):
                self.assertEqual(self.client.post('/match', json={'names': ['Panadol']}).status_code, 200)
            self.assertEqual(self.client.post('/match', json={'names': ['Panadol']}).status_code, 429)
        with patch.dict(os.environ, {'GEMINI_CALLS_PER_MINUTE': '2'}):
            gemini_runtime.reserve_call()
            gemini_runtime.reserve_call()
            with self.assertRaises(HTTPException) as error:
                gemini_runtime.reserve_call()
            self.assertEqual(error.exception.status_code, 429)


if __name__ == '__main__':
    unittest.main()
