import { expect, test } from "vitest";
import { deriv, numpy as np } from "jax-js";

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

test("can create array", async () => {
  // const result = np.neg(np.cos(np.array([1, 2, 3])));
  // np.debugPrint(result);

  // const [y, sinderiv] = np.jvpV1(np.sin, [x], [1.0]);
  // console.log(await y.js());
  // console.log(await sinderiv.js());

  const x = 3.0;

  console.log(np.sin(x).js());
  console.log(deriv(np.sin)(x).js());
  console.log(deriv(deriv(np.sin))(x).js());
  console.log(deriv(deriv(deriv(np.sin)))(x).js());
});
