// Mirrors the `jax.scipy.special` module in JAX.

import { fudgeArray } from "./frontend/array";
import * as core from "./frontend/core";
import { jit } from "./frontend/jaxpr";
import { Array, ArrayLike, log, subtract } from "./numpy";

/** The error function: `erf(x) = 2/sqrt(pi) * int[0..x] exp(-t^2) dt`. */
export function erf(x: ArrayLike): Array {
  return core.erf(x) as Array;
}

/**
 * The complementary error function: `erfc(x) = 1 - erf(x)`.
 *
 * This function is more accurate than `1 - erf(x)` for large values of `x`,
 * where `erf(x)` is very close to 1.
 */
export function erfc(x: ArrayLike): Array {
  return core.erfc(x) as Array;
}

export { logSoftmax } from "./nn";

/**
 * @function
 * The logit function, `logit(p) = log(p / (1-p))`.
 */
export const logit = jit(function logit(x: ArrayLike): Array {
  x = fudgeArray(x);
  return log(x.ref.div(subtract(1, x)));
});

export { logsumexp } from "./nn";
export { softmax } from "./nn";
