"""Assemble FG Plate Lab from its parts.

Run from the project root (after experiments/export_demo_model.py):
    python demo/build_demo.py

Writes
  demo/fg_plate_lab.html          page body for publishing as a claude.ai artifact (Plotly from the jsDelivr CDN)
  demo/fg_plate_lab_offline.html  complete standalone file with Plotly inlined; open it in any browser, no internet needed
                                  (fonts fall back to system fonts when offline)
  demo/web/index.html             complete web page (Plotly from the CDN) for GitHub Pages or the Streamlit wrapper
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLOTLY_TAG = '<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>'
VENDOR = HERE/'vendor'/'plotly-2.35.2.min.js'


def safe(js):
    return js.replace('</script', '<\\/script')


def main():
    src = (HERE/'fg_plate_lab.src.html').read_text(encoding='utf-8')
    data = (HERE/'fg_plate_lab_data.json').read_text(encoding='utf-8')
    math = (HERE/'fg_plate_math.js').read_text(encoding='utf-8')
    assert src.count('/*DATA*/') == 1 and src.count('/*MATH*/') == 1 and src.count(PLOTLY_TAG) == 1
    page = src.replace('/*DATA*/', safe(data)).replace('/*MATH*/', safe(math))
    (HERE/'fg_plate_lab.html').write_text(page, encoding='utf-8')
    plotly = VENDOR.read_text(encoding='utf-8')
    offline = page.replace(PLOTLY_TAG, '<script>'+safe(plotly)+'</script>')
    offline = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
               '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
               '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
               'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
               + offline + '\n</body>\n</html>\n')
    (HERE/'fg_plate_lab_offline.html').write_text(offline, encoding='utf-8')
    # web/index.html: complete document for any static host (GitHub Pages, Netlify) or the Streamlit wrapper
    head = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<style>:root{padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
            'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n')
    (HERE/'web').mkdir(exist_ok=True)
    (HERE/'web'/'index.html').write_text(head + page + '\n</body>\n</html>\n', encoding='utf-8')
    for name in ('fg_plate_lab.html', 'fg_plate_lab_offline.html', 'web/index.html'):
        print(f'wrote demo/{name} ({(HERE/name).stat().st_size/1e6:.2f} MB)')


if __name__ == '__main__':
    main()
