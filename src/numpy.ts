import { DType } from "./alu";
import {
  array,
  Array,
  eye,
  full,
  ones,
  pureArray,
  scalar,
  zeros,
} from "./frontend/array";
import * as core from "./frontend/core";
import * as vmapModule from "./frontend/vmap";
import { deepEqual, range } from "./utils";

export { Array, array, DType, eye, scalar, zeros, ones, full };

export const float32 = DType.Float32;
export const int32 = DType.Int32;
export const bool = DType.Bool;
export const complex64 = DType.Complex64;

// Constants section

/** Euler's constant, `e = 2.7182818284590...` */
export const e = Math.E;

/** Euler-Mascheroni constant, `γ = 0.5772156649... */
export const eulerGamma = 0.5772156649015329;

/** Positive infinity. */
export const inf = Number.POSITIVE_INFINITY;

/** Floating-point representation of NaN. */
export const nan = Number.NaN;

/** This is Pi, `π = 3.14159265358979...` */
export const pi = Math.PI;

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
export const equal = core.equal as (x: ArrayLike, y: ArrayLike) => Array;
export const notEqual = core.notEqual as (x: ArrayLike, y: ArrayLike) => Array;
export const greaterEqual = core.greaterEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
export const lessEqual = core.lessEqual as (
  x: ArrayLike,
  y: ArrayLike,
) => Array;
export const where = core.where as (
  cond: ArrayLike,
  x: ArrayLike,
  y: ArrayLike,
) => Array;
export const transpose = core.transpose as (
  x: ArrayLike,
  perm?: number[],
) => Array;
export const broadcast = core.broadcast as (
  x: ArrayLike,
  shape: number[],
  axis: number[],
) => Array;
export const sum = core.reduceSum as (
  x: ArrayLike,
  axis?: number | number[],
) => Array;
export const moveaxis = vmapModule.moveaxis as (
  x: ArrayLike,
  src: number,
  dst: number,
) => Array;

// Version of pureArray with fudged types.
const fudgeArray = pureArray as (x: ArrayLike) => Array;

/**
 * Return specified diagonals.
 *
 * If a is 2D, return the diagonal of the array with the given offset. If a is
 * 3D or higher, compute diagonals along the two given axes.
 *
 * This returns a view over the existing array.
 */
export function diagonal(
  a: ArrayLike,
  offset?: number,
  axis1?: number,
  axis2?: number,
): Array {
  return fudgeArray(a).diagonal(offset, axis1, axis2);
}

/** Transposes a matrix or stack of matrices `x` (swap last two axes). */
export function matrixTranspose(x: ArrayLike): Array {
  const ar = fudgeArray(x);
  if (ar.ndim < 2)
    throw new TypeError("matrixTranspose only supports 2D+ arrays");
  return ar.transpose([...range(ar.ndim - 2), ar.ndim - 1, ar.ndim - 2]);
}

/**
 * Extract a diagonal or construct a diagonal array.
 *
 * If v is a 2D array, return the k-th diagonal of v (as a view). If v is a 1D
 * array, return a 2D array with v on the k-th diagonal.
 */
export function diag(v: ArrayLike, k = 0): Array {
  const a = fudgeArray(v);
  if (!Number.isInteger(k))
    throw new TypeError(`k must be an integer, got ${k}`);
  if (a.ndim === 1) {
    const n = a.shape[0];
    const ret = where(eye(n).equal(1), a, 0);
    // TODO: pad() is unimplemented at this layer
    if (k !== 0) throw new Error("diag() for 1D arrays only for k=0");
    return ret;
  } else if (a.ndim === 2) {
    return diagonal(a, k);
  } else {
    throw new TypeError("numpy.diag only supports 1D and 2D arrays");
  }
}

/** Compute the number of dimensions of an array. */
export const ndim = core.ndim as (x: ArrayLike) => number;

/** Return if two arrays are element-wise equal within a tolerance. */
export function allclose(
  actual: Parameters<typeof array>[0],
  expected: Parameters<typeof array>[0],
  options?: { rtol?: number; atol?: number },
): boolean {
  const { rtol = 1e-5, atol = 1e-8 } = options ?? {};

  const x = array(actual);
  const y = array(expected);
  if (!deepEqual(x.shape, y.shape)) {
    return false;
  }
  const xData = x.dataSync();
  const yData = y.dataSync();
  for (let i = 0; i < xData.length; i++) {
    if (Math.abs(xData[i] - yData[i]) > atol + rtol * Math.abs(yData[i])) {
      return false;
    }
  }
  return true;
}
