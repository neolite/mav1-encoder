#!/usr/bin/env python3
"""Extract constants and tables from the AV1 spec markdown into a C header.

usage: gen_spec_tables.py <av1-spec checkout> <out.h>
"""
import re, sys, glob, os

spec, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(spec, '0*.md')) +
               glob.glob(os.path.join(spec, '10*.md')))

consts = []
const_re = re.compile(r'\|\s*`?([A-Z][A-Z0-9_]+)`?\s*\|\s*(-?[0-9A-Z_\\<*() -]+?)\s*\|')
for line in open(os.path.join(spec, '03.symbols.md')):
    m = const_re.match(line)
    if m:
        consts.append((m.group(1), '(' + m.group(2).replace('\\', '') + ')'))

table_re = re.compile(r'^\s*([A-Z][A-Za-z0-9_]*)((?:\s*\[[^\]]*\])+)\s*=\s*\{', re.M)
tables = {}
for f in files:
    txt = open(f).read()
    for block in re.findall(r'~~~~~[^\n]*\n(.*?)~~~~~', txt, re.S):
        for m in table_re.finditer(block):
            name = m.group(1)
            i = m.end() - 1
            depth = 0
            for j in range(i, len(block)):
                if block[j] == '{':
                    depth += 1
                elif block[j] == '}':
                    depth -= 1
                    if depth == 0:
                        break
            body = block[i:j + 1]
            if name in tables or re.search(r'[a-z]\w*\s*\(', body):
                continue
            # The spec source occasionally omits a separator at a line end.
            body = re.sub(r'(\w)([ \t]*\n\s*)(?=\w)', r'\1,\2', body)
            tables[name] = (m.group(2), body)

with open(out, 'w') as o:
    o.write('/* Generated from the AV1 specification (AOMedia, BSD-2) by '
            'gen_spec_tables.py. Do not edit. */\n')
    o.write('#ifndef SPEC_TABLES_H\n#define SPEC_TABLES_H\n'
            '#ifdef __METAL_VERSION__\n#define SPEC_CONST constant\n#else\n'
            '#include <stdint.h>\n#define SPEC_CONST static const\n#endif\n')
    for n, v in consts:
        o.write(f'#define {n} {v}\n')
    o.write('#include "spec_enums.h"\n')
    for n, (dims, body) in tables.items():
        typ = 'uint16_t' if n.endswith('Cdf') else 'int32_t'
        o.write(f'SPEC_CONST {typ} {n}{dims} = {body};\n')
    o.write('#endif\n')
print(len(consts), 'consts', len(tables), 'tables')
