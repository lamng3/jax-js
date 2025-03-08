import { test } from "vitest";
import { getBackend, BackendOp } from "../backend";

test("eric", async () => {
  const backend = await getBackend("webgpu");

  const a = backend.malloc(3 * 4, new Float32Array([1, 2, 3]).buffer);
  const b = backend.malloc(3 * 4, new Float32Array([4, 5, 6]).buffer);
  const c = backend.malloc(3 * 4);

  try {
    await backend.executeOp(BackendOp.Mul, [a, b], [c]);
    const buf = await backend.read(c);
    console.log("result:", new Float32Array(buf));
  } finally {
    backend.decRef(a);
    backend.decRef(b);
    backend.decRef(c);
  }
});
