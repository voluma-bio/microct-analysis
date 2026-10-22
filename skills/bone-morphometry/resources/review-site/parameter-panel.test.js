import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { resolveParameter, formatRelated } from "./parameter-panel.js";
const catalog = JSON.parse(
  readFileSync(new URL("./parameters.json", import.meta.url)),
);
const data = {
  measurements: {
    femoral_width_provisional: {
      value: 3.393,
      unit: "mm",
      points_scene_mm: [
        [0, 0, 0],
        [3.393, 0, 0],
      ],
    },
  },
  structures: [{ label: 6, volume_mm3: 0.41 }],
};
test("all presentation parameters and vertebral counterparts are covered", () => {
  assert.deepEqual(
    catalog.parameters.map((p) => p.id),
    Array.from({ length: 23 }, (_, i) => i + 1),
  );
  assert.deepEqual(
    catalog.vertebral.parameter_ids,
    [15, 16, 17, 18, 19, 20, 21, 22, 23],
  );
});
test("prior width retains its explicit definition mismatch", () => {
  const p = catalog.parameters.find((p) => p.id === 5),
    resolved = resolveParameter(p, data);
  assert.equal(resolved.parameter.status, "Definition mismatch");
  assert.equal(resolved.value, 3.393);
  assert.match(formatRelated(p, data), /prior estimate/);
});
test("unmeasured tibial and ROI parameters never inherit unrelated results", () => {
  for (const id of [
    7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
  ]) {
    const p = catalog.parameters.find((p) => p.id === id),
      resolved = resolveParameter(p, data);
    assert.equal(resolved.prior_value, null);
    assert.equal(resolved.value, undefined);
    assert.equal(resolved.points_scene_mm, undefined);
    assert.equal(formatRelated(p, data), "Not measured");
  }
});
test("mineralized candidate volume is not promoted to measured meniscus volume", () => {
  const p = catalog.parameters.find((p) => p.id === 2),
    resolved = resolveParameter(p, data);
  assert.equal(resolved.prior_value, 0.41);
  assert.equal(resolved.value, undefined);
  assert.match(formatRelated(p, data), /candidate label/);
});
