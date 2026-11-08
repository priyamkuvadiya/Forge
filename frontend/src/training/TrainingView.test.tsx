import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { json, wire } from "../test/wire";
import { TrainingView } from "./TrainingView";

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("training dashboard", () => {
  it("charts a real run excerpt, counts its update-free step, and leaves unlogged fields empty", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => (url === "/api/runs" ? json({ runs: wire.runs }) : json(wire.run))),
    );
    render(<TrainingView />);
    const row = (await screen.findByText("Correctness-only reward. Served as run1.")).closest("tr")!;
    const cells = [...row.querySelectorAll("td")].map((td) => td.textContent);
    const idle = wire.run.records.filter((r) => r.kept_episodes === 0).length;
    expect(idle).toBe(1);
    // run, steps, lr, penalty, categories, no-update steps, spilled, ...
    expect(cells.slice(1, 7)).toEqual([String(wire.run.steps), "0.00001", "none", "all five", String(idle), "0"]);

    // run1 never logged the penalty fields: those charts say so instead of drawing zeros.
    const empty = screen.getAllByText("Not logged by the selected runs.");
    expect(empty.length).toBe(2);
    const titles = empty.map((node) => node.closest("figure")!.querySelector("strong")!.textContent);
    expect(titles).toEqual(["Penalised episodes", "Episodes that gave an answer"]);
  });
});
