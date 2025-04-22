import "vitest";
import { numpy as np } from "@jax-js/core";

interface CustomMatchers<R = unknown> {
  toBeAllclose: (expected: Parameters<typeof np.array>[0]) => R;
}

declare module "vitest" {
  interface Assertion<T = any> extends CustomMatchers<T> {}
  interface AsymmetricMatchersContaining extends CustomMatchers {}
}
