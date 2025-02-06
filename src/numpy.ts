import * as tf from "@tensorflow/tfjs-core";

import { Array, DType } from "./core";
import * as core from "./core";

export { Array, DType };

export const float32 = DType.Float32;
export const int32 = DType.Int32;
export const bool = DType.Bool;
export const complex64 = DType.Complex64;

// Note: These primitive wrappers have fudged types.
//
// They can take any `TracerValue` and return any `Tracer` subclass based on the
// current stack of interpreters. But we hide that away from users to mimic
// JAX's composable tracing transformations.

export type ArrayLike = Array | number | boolean;

export const add = core.add as (x: ArrayLike, y: ArrayLike) => Array;
export const mul = core.mul as (x: ArrayLike, y: ArrayLike) => Array;
export const neg = core.neg as (x: ArrayLike) => Array;
export const sin = core.sin as (x: ArrayLike) => Array;
export const cos = core.cos as (x: ArrayLike) => Array;
export const greater = core.greater as (x: ArrayLike, y: ArrayLike) => Array;
export const less = core.less as (x: ArrayLike, y: ArrayLike) => Array;
export const transpose = core.transpose as (
  x: ArrayLike,
  perm?: number[]
) => Array;
export const broadcast = core.broadcast as (
  x: ArrayLike,
  shape: number[],
  axes: number[]
) => Array;
export const reduceSum = core.reduceSum as (
  x: ArrayLike,
  axis?: number | number[]
) => Array;

export function array(
  values: tf.TensorLike,
  { shape, dtype }: { shape?: number[]; dtype?: DType } = {}
): Array {
  return new Array(tf.tensor(values, shape, dtype));
}
