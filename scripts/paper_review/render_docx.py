"""Render with native Windows Word and existing PyMuPDF; QA artifacts only."""
from pathlib import Path
import json
import argparse
import subprocess
import sys
import pymupdf as fitz

ROOT=Path(__file__).resolve().parents[2]
DOC=ROOT/'background/EI论文_前两部分修订版.docx'
OUT=ROOT/'artifacts/paper_review_20261103/render'

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,default=DOC)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args();destination=args.output_dir.resolve()
    destination.mkdir(parents=True,exist_ok=True)
    pdf=destination/'review.pdf'
    command=['powershell.exe','-NoProfile','-NonInteractive','-File',str(Path(__file__).with_name('export_word_pdf.ps1')),'-InputPath',str(args.input.resolve()),'-OutputPath',str(pdf)]
    result=subprocess.run(command,capture_output=True,timeout=150)
    (destination/'word_export.log').write_bytes(result.stdout+result.stderr)
    if result.returncode:
        sys.stderr.write((result.stdout+result.stderr).decode('utf-8',errors='replace'));raise SystemExit(result.returncode)
    pages=[]
    with fitz.open(pdf) as document:
        for i,page in enumerate(document):
            png=destination/f'page-{i+1}.png';page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(png)
            pages.append({'page':i+1,'image':str(png.relative_to(ROOT)),'text_characters':len(page.get_text()),'start':page.get_text()[:80],'end':page.get_text()[-80:]})
        (destination/'rendered_text.txt').write_text('\n\f\n'.join(p.get_text() for p in document),encoding='utf-8')
    (destination/'pages.json').write_text(json.dumps(pages,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(pages,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
