import * as numpy from "./numpy";
import { Array, ArrayLike } from "./numpy";
import * as core from "./core";

export { numpy };

// Fudged array types for composable transformations.
export const jvpV1 = core.jvpV1 as unknown as (
  f: (x: ArrayLike) => Array,
  primals: ArrayLike[],
  tangents: ArrayLike[]
) => [Array, Array];
export const deriv = core.deriv as unknown as (
  f: (x: ArrayLike) => Array
) => (x: ArrayLike) => Array;
