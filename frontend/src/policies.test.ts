import { describe, expect, it } from "vitest";
import { labelFor, orderPolicies, TRAINED } from "./policies";

describe("policies", () => {
  it("run2 is the one called trained", () => {
    expect(TRAINED).toBe("run2");
    expect(labelFor("run2").title).toMatch(/^Trained/);
  });

  it("baseline first, then trained, then the rest by name", () => {
    const names = orderPolicies([{ name: "run1" }, { name: "zeta" }, { name: "run2" }, { name: "baseline" }]).map(
      (p) => p.name,
    );
    expect(names).toEqual(["baseline", "run2", "run1", "zeta"]);
  });

  it("an adapter it does not know is shown by name, not hidden", () => {
    expect(labelFor("run3").title).toBe("run3");
  });
});
