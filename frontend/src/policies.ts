// What the demo calls each policy. The backend serves whatever adapters it was
// started with; these are the names the README's commands give them.
//
// "Trained" is run2: it scores higher than run1 held-out (macro 0.503 vs 0.467)
// and on four of five categories. run1 stays selectable because it is the one
// that chains search into the calculator on multi_tool; run2 never calls the
// calculator there (0 of 120 held-out episodes, against run1's 63), it searches
// and does the arithmetic in its head.

export const TRAINED = "run2";

export interface PolicyLabel {
  title: string;
  note: string;
}

const labels: Record<string, PolicyLabel> = {
  baseline: {
    title: "Baseline",
    note: "Qwen2.5-0.5B-Instruct, prompted, no training. Held-out macro 0.183.",
  },
  run2: {
    title: "Trained (run2)",
    note:
      "GRPO with the no-search penalty. Held-out macro 0.503. Searches on QA, " +
      "but never calls the calculator on multi_tool: it adds in its head.",
  },
  run1: {
    title: "run1",
    note:
      "GRPO, correctness-only reward. Held-out macro 0.467. Still chains search " +
      "into the calculator on multi_tool, and answers QA without searching.",
  },
};

export function labelFor(name: string): PolicyLabel {
  return labels[name] ?? { title: name, note: "An adapter this page has no description for." };
}

/** Baseline first, then trained, then anything else by name. */
export function orderPolicies<T extends { name: string }>(policies: T[]): T[] {
  const rank = (name: string) => (name === "baseline" ? 0 : name === TRAINED ? 1 : 2);
  return [...policies].sort((a, b) => rank(a.name) - rank(b.name) || a.name.localeCompare(b.name));
}
