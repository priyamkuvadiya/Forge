import { useId, useMemo, useState } from "react";
import type { Point } from "./series";

export interface Line {
  label: string;
  color: string;
  points: Point[];
  /** Raw per-step values drawn faintly behind the line, if given. */
  raw?: Point[];
}

interface Props {
  title: string;
  lines: Line[];
  yDomain?: [number, number];
  format?: (value: number) => string;
  note?: string;
  height?: number;
}

const WIDTH = 520;
const PAD = { top: 10, right: 12, bottom: 24, left: 44 };

/** A small hand-drawn SVG line chart. Null values are gaps, never zeros. */
export function LineChart({ title, lines, yDomain, format = (v) => v.toFixed(2), note, height = 180 }: Props) {
  const clip = useId();
  const [hover, setHover] = useState<number | null>(null);

  const { xMin, xMax, yMin, yMax } = useMemo(() => {
    const all = lines.flatMap((l) => [...l.points, ...(l.raw ?? [])]);
    const steps = all.map((p) => p.step);
    const values = all.map((p) => p.value).filter((v): v is number => v !== null);
    let lo = yDomain?.[0] ?? Math.min(0, ...values);
    let hi = yDomain?.[1] ?? Math.max(...values, lo + 1e-9);
    if (!yDomain) {
      const span = hi - lo || 1;
      hi += span * 0.05;
      if (lo < 0) lo -= span * 0.05;
    }
    return {
      xMin: steps.length ? Math.min(...steps) : 0,
      xMax: steps.length ? Math.max(...steps) : 1,
      yMin: lo,
      yMax: hi,
    };
  }, [lines, yDomain]);

  const innerW = WIDTH - PAD.left - PAD.right;
  const innerH = height - PAD.top - PAD.bottom;
  const x = (step: number) => PAD.left + ((step - xMin) / Math.max(1, xMax - xMin)) * innerW;
  const y = (value: number) => PAD.top + (1 - (value - yMin) / (yMax - yMin || 1)) * innerH;

  const path = (points: Point[]) => {
    let d = "";
    let pen = false;
    for (const p of points) {
      if (p.value === null) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${x(p.step).toFixed(1)},${y(p.value).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  const ticks = [yMin, (yMin + yMax) / 2, yMax];
  const xTicks = [xMin, Math.round((xMin + xMax) / 2), xMax];
  const empty = lines.every((l) => l.points.every((p) => p.value === null));

  const onMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    const px = ((event.clientX - box.left) / box.width) * WIDTH;
    const step = Math.round(xMin + ((px - PAD.left) / innerW) * (xMax - xMin));
    setHover(Math.min(xMax, Math.max(xMin, step)));
  };

  const at = (points: Point[], step: number) => points.find((p) => p.step === step)?.value ?? null;

  return (
    <figure className="chart">
      <figcaption>
        <strong>{title}</strong>
        <span className="legend">
          {lines.map((l) => (
            <span key={l.label}>
              <i style={{ background: l.color }} />
              {l.label}
              {hover !== null && (
                <span className="muted">
                  {" "}
                  {(() => {
                    const v = at(l.points, hover);
                    return v === null ? "–" : format(v);
                  })()}
                </span>
              )}
            </span>
          ))}
          {hover !== null && <span className="muted">step {hover}</span>}
        </span>
      </figcaption>
      {empty ? (
        <div className="chart-empty muted">Not logged by the selected runs.</div>
      ) : (
        <svg
          viewBox={`0 0 ${WIDTH} ${height}`}
          role="img"
          aria-label={title}
          onMouseMove={onMove}
          onMouseLeave={() => setHover(null)}
        >
          <defs>
            <clipPath id={clip}>
              <rect x={PAD.left} y={PAD.top} width={innerW} height={innerH} />
            </clipPath>
          </defs>
          {ticks.map((t) => (
            <g key={t}>
              <line className="grid" x1={PAD.left} x2={WIDTH - PAD.right} y1={y(t)} y2={y(t)} />
              <text className="axis" x={PAD.left - 6} y={y(t) + 4} textAnchor="end">
                {format(t)}
              </text>
            </g>
          ))}
          {xTicks.map((t) => (
            <text key={t} className="axis" x={x(t)} y={height - 6} textAnchor="middle">
              {t}
            </text>
          ))}
          <g clipPath={`url(#${clip})`}>
            {lines.map(
              (l) =>
                l.raw && <path key={`${l.label}-raw`} d={path(l.raw)} stroke={l.color} className="raw-line" fill="none" />,
            )}
            {lines.map((l) => (
              <path key={l.label} d={path(l.points)} stroke={l.color} className="main-line" fill="none" />
            ))}
          </g>
          {hover !== null && (
            <line className="cursor" x1={x(hover)} x2={x(hover)} y1={PAD.top} y2={PAD.top + innerH} />
          )}
        </svg>
      )}
      {note && <p className="chart-note muted">{note}</p>}
    </figure>
  );
}
