import { expect, test } from "vitest";
import { numpy as np } from "jax-js";

// test("x is 3", () => {
//   expect(x).toBe(3);
// });

// test("has webgpu", async () => {
//   const adapter = await navigator.gpu?.requestAdapter();
//   const device = await adapter?.requestDevice();
//   if (!adapter || !device) {
//     throw new Error("No adapter or device");
//   }
//   console.log(device.adapterInfo.architecture);
//   console.log(device.adapterInfo.vendor);
//   console.log(adapter.limits.maxVertexBufferArrayStride);
// });

test("can create array", () => {
  // const result = np.neg(np.cos(np.array([1, 2, 3])));
  // np.debugPrint(result);

  const x = 3.0;
  const [y, sinderiv] = np.jvpV1(np.sin, [x], [1.0]);
  np.debugPrint(y);
  np.debugPrint(sinderiv);
});
