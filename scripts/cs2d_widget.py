"""Docs-widget code for the 2D surface of ts_cs2d.py: a map of the energy over
the hydride coordinate and the C6-S distance, with the minimax path and its
highest point. Prints the one-line module to stdout.

    python scripts/cs2d_widget.py oniom|xtb ["title words"] > widget.tsx
"""
import heapq, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K = 627.509474


def rows_for(level):
    d = json.loads((ROOT / 'results' / 'ts_cs2d.json').read_text())
    P, XI, R = d['points'], d['config']['xi'], d['config']['rcs']
    k0 = f'{XI[0]:+.1f}_{R[0]:.2f}'
    E0 = P[k0][f'E_{level}']
    E = {(i, j): (P[f'{x:+.1f}_{r:.2f}'][f'E_{level}'] - E0) * K
         for i, x in enumerate(XI) for j, r in enumerate(R)
         if f'E_{level}' in P.get(f'{x:+.1f}_{r:.2f}', {})}
    # minimax path from the reactant corner to the lowest point of the last xi column reached
    best, parent, heap = {(0, 0): E[(0, 0)]}, {}, [(E[(0, 0)], 0, 0)]
    while heap:
        h, i, j = heapq.heappop(heap)
        if h > best[(i, j)]:
            continue
        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
            if (a, b) in E and max(h, E[(a, b)]) < best.get((a, b), float('inf')):
                best[(a, b)] = max(h, E[(a, b)]); parent[(a, b)] = (i, j)
                heapq.heappush(heap, (best[(a, b)], a, b))
    last = max(i for i, _ in E)
    end = min(((last, j) for j in range(len(R)) if (last, j) in E), key=lambda c: E[c])
    path, c = [end], end
    while c != (0, 0):
        c = parent[c]; path.append(c)
    step = {c: n for n, c in enumerate(reversed(path))}
    return [dict(xi=XI[i], r=R[j], e=round(e, 2), step=step.get((i, j), -1)) for (i, j), e in sorted(E.items())]


def module(rows, title):
    data = ','.join(f"{{xi:{r['xi']}, r:{r['r']}, e:{r['e']}, step:{r['step']}}}" for r in rows)
    return ("export default () => <claude.Visualize data-claude-component='cs-surface' sources={{rows:{kind:'data', data:["
            + data + "]}}}>{({rows, datum}) => { const xs = [...new Set(rows.map(r => r.xi))].sort((a, b) => a - b), "
            "rs = [...new Set(rows.map(r => r.r))].sort((a, b) => a - b); const x0 = 96, x1 = 600, y0 = 100, y1 = 412; "
            "const cw = (x1 - x0) / xs.length, ch = (y1 - y0) / rs.length; const X = v => x0 + xs.indexOf(v) * cw, "
            "Y = v => y1 - (rs.indexOf(v) + 1) * ch; const emax = Math.max(...rows.map(r => r.e)), "
            "op = e => 0.06 + 0.84 * Math.max(0, Math.min(1, e / emax)); const path = rows.filter(r => r.step >= 0)"
            ".sort((a, b) => a.step - b.step); const peak = path.reduce((m, r) => r.e > m.e ? r : m); "
            "const end = path.at(-1); const accent = 'var(--cds-chart-categorical-1)', ink = 'var(--cds-text-primary)', "
            "quiet = 'var(--cds-text-secondary)'; const ticks = [0, 0.25, 0.5, 0.75, 1].map(f => f * emax); "
            "return <svg viewBox='0 0 760 482' role='img' aria-label='" + title + "' fontSize='12'>"
            "<text data-claude-text-id='title' x='20' y='30' fontSize='15' fontWeight='600' fill={ink}>" + title + "</text>"
            "<text data-claude-text-id='subtitle' x='20' y='52' fontSize='12' fill={quiet}>Energy relative to the "
            "reactant, kcal/mol; line: lowest-barrier path across the grid</text>"
            "<g data-claude-anchor='cells'>{rows.map(r => <rect key={r.xi + '_' + r.r} x={X(r.xi)} y={Y(r.r)} "
            "width={cw - 1} height={ch - 1} fill={accent} fillOpacity={op(r.e)} {...datum(r, 'e')}><title>"
            "{`ξ ${r.xi.toFixed(1)} Å, C6–S ${r.r.toFixed(2)} Å: ${r.e.toFixed(1)} kcal/mol`}"
            "</title></rect>)}</g><g data-claude-anchor='path'><polyline points={path.map(r => `${X(r.xi) + cw / 2},"
            "${Y(r.r) + ch / 2}`).join(' ')} fill='none' stroke={ink} strokeWidth='2'/>{path.map(r => <circle "
            "key={r.step} cx={X(r.xi) + cw / 2} cy={Y(r.r) + ch / 2} r='2.5' fill={ink}/>)}</g>"
            "<g data-claude-anchor='saddle'><circle cx={X(peak.xi) + cw / 2} cy={Y(peak.r) + ch / 2} r='8' fill='none' "
            "stroke={ink} strokeWidth='2' {...datum(peak, 'e')}/></g><g data-claude-anchor='ends'>"
            "<rect x={X(end.xi) + cw / 2 - 6} y={Y(end.r) + ch / 2 - 6} width='12' height='12' fill='none' stroke={ink} "
            "strokeWidth='2' {...datum(end, 'e')}/><text data-claude-text-id='reactant' "
            "x={x0 + 4} y={y1 + 32} fontSize='11.5' fill={quiet}>reactant</text></g>"
            "<text x='20' y='74' fontSize='12' fontWeight='600' fill={ink} {...datum(peak, 'e')}>{`\u25cb highest point on the "
            "path: ${peak.e.toFixed(1)} kcal/mol at \u03be = ${peak.xi.toFixed(1)} \u00c5, C6\u2013S = ${peak.r.toFixed(2)} "
            "\u00c5    \u25a1 end of path: ${end.e >= 0 ? '+' : ''}${end.e.toFixed(1)} kcal/mol`}</text><g data-claude-anchor='x-axis'>"
            "{xs.filter((v, i) => i % 2 === 0).map(v => <text key={v} x={X(v) + cw / 2} y={y1 + 16} "
            "textAnchor='middle' fontSize='11' fill={quiet}>{v.toFixed(1)}</text>)}<text data-claude-text-id='x-label' "
            "x={(x0 + x1) / 2} y={y1 + 50} textAnchor='middle' fontSize='11.5' fill={quiet}>hydride coordinate "
            "ξ = d(C6–H) − d(H–C11), Å</text></g><g data-claude-anchor='y-axis'>"
            "{rs.map(v => <text key={v} x={x0 - 8} y={Y(v) + ch / 2 + 4} textAnchor='end' fontSize='11' fill={quiet}>"
            "{v.toFixed(2)}</text>)}<text data-claude-text-id='y-label' x='24' y={(y0 + y1) / 2} "
            "transform={`rotate(-90 24 ${(y0 + y1) / 2})`} textAnchor='middle' fontSize='11.5' fill={quiet}>"
            "C6–S(Cys146) distance, Å</text></g><g data-claude-anchor='legend'>{ticks.map(t => <g key={t}>"
            "<rect x='640' y={y1 - (t / emax) * (y1 - y0) - 8} width='18' height='16' fill={accent} "
            "fillOpacity={op(t)}/><text x='666' y={y1 - (t / emax) * (y1 - y0) + 4} fontSize='11' fill={quiet}>"
            "{t.toFixed(0)}</text></g>)}<text data-claude-text-id='legend-unit' x='640' y={y0 - 14} fontSize='11' "
            "fill={quiet}>kcal/mol</text></g></svg>; }}</claude.Visualize>;")


if __name__ == '__main__':
    level = sys.argv[1] if len(sys.argv) > 1 else 'oniom'
    title = sys.argv[2] if len(sys.argv) > 2 else 'Energy over the hydride coordinate and the C6–S distance'
    print(module(rows_for(level), title))
