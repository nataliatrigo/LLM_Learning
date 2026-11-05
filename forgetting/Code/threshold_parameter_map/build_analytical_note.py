"""Export this note's Markdown subset to a standalone, compilable LaTeX file."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/threshold_parameter_map' / 'analytical_notes'


def escape(text):
    replacements = {'&': r'\&', '%': r'\%', '#': r'\#', '_': r'\_',
                    '{': r'\{', '}': r'\}', '~': r'\textasciitilde{}'}
    return ''.join(replacements.get(char, char) for char in text)


def inline(text):
    parts = re.split(r'(\$[^$\n]+\$|`[^`]+`|\*\*[^*]+\*\*)', text)
    output = []
    for part in parts:
        if part.startswith('$'):
            output.append(part)
        elif part.startswith('`'):
            output.append(r'\path{' + part[1:-1] + '}')
        elif part.startswith('**'):
            output.append(r'\textbf{' + inline(part[2:-2]) + '}')
        else:
            output.append(escape(part))
    return ''.join(output)


def main():
    lines = (ROOT / 'analytical_regions.md').read_text().splitlines()
    output = [r'''\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage[margin=22mm]{geometry}
\usepackage{amsmath,amssymb,mathtools}
\usepackage{tabularx,booktabs}
\usepackage{xurl}
\usepackage[hidelinks]{hyperref}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{2em}
\allowdisplaybreaks[1]
\title{Analytical information about the policy phase regions}
\author{}
\date{}
\begin{document}
\maketitle
''']
    index = 1
    while index < len(lines):
        line = lines[index]
        if not line.strip():
            output.append('')
            index += 1
        elif line == '$$':
            index += 1
            equation = []
            while lines[index] != '$$':
                equation.append(lines[index])
                index += 1
            output.append('\\[\n' + '\n'.join(equation) + '\n\\]')
            index += 1
        elif line.startswith('### '):
            output.append(r'\subsection*{' + inline(line[4:]) + '}')
            index += 1
        elif line.startswith('## '):
            output.append(r'\section*{' + inline(line[3:]) + '}')
            index += 1
        elif line.startswith(('* ', '- ')):
            output.append(r'\begin{itemize}')
            while index < len(lines) and lines[index].startswith(('* ', '- ')):
                output.append(r'\item ' + inline(lines[index][2:]))
                index += 1
            output.append(r'\end{itemize}')
        elif line.startswith('|'):
            output.append(r'\begin{center}\small\begin{tabularx}{\linewidth}{@{}llX@{}}\toprule')
            header = True
            while index < len(lines) and lines[index].startswith('|'):
                cells = [cell.strip() for cell in lines[index].strip('|').split('|')]
                if not all(re.fullmatch(r'[-:]+', cell) for cell in cells):
                    output.append(' & '.join(inline(cell) for cell in cells) + r' \\')
                    if header:
                        output.append(r'\midrule')
                        header = False
                index += 1
            output.append(r'\bottomrule\end{tabularx}\end{center}')
        else:
            output.append(inline(line))
            index += 1
    output.append(r'\end{document}')
    target = ROOT / 'analytical_regions.tex'
    document = '\n'.join(output) + '\n'
    # Keep each introductory sentence with its display instead of starting a new paragraph.
    document = re.sub(r'\n\n+(?=\\\[)', '\n', document)
    target.write_text(document)
    print(target)


if __name__ == '__main__':
    main()
