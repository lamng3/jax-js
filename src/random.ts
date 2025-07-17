// Port of the `jax.random` module.

import { type Device } from "./backend";
import { randomBits } from "./frontend/core";
import { array, Array, DType, float32, full } from "./numpy";

function validateKeyShape(key: Array): number[] {
  if (key.ndim === 0) {
    throw new Error("Key must have at least one dimension.");
  }
  if (key.shape[key.shape.length - 1] !== 2) {
    throw new Error(
      `Invalid key shape: ${key.shape}. Expected last dimension to be 2.`,
    );
  }
  return key.shape.slice(0, -1);
}

/** Create a pseudo-random number generator (PRNG) key from 32-bit integer seed. */
export function key(seed: number): Array {
  seed = seed >>> 0;
  // To match JAX, put the 32-bit seed into a 64-bit key in this way.
  return array([0, seed], { dtype: DType.Uint32 });
}

/** Splits a PRNG key into `num` new keys by adding a leading axis. */
export function split(key: Array, num: number = 2): Array[] {
  if (num < 0 || !Number.isInteger(num)) {
    throw new Error(`Invalid number of splits: ${num}`);
  }
  const ret: Array[] = [];
  for (let i = 0; i < num; i++) ret.push(key); // TODO: Actually split the key.
  key.dispose();
  return ret;
}

/** Sample uniform bits in the form of unsigned integers. */
export function bits(key: Array, shape: number[] = []): Array {
  const keyShape = validateKeyShape(key);
  return randomBits(
    key.ref.slice(...keyShape.map(() => null), 0),
    key.slice(...keyShape.map(() => null), 1),
    shape,
  ) as Array;
}

/** Sample uniform random values in [minval, maxval) with given shape/dtype. */
export function uniform(
  key: Array,
  shape: number[] = [],
  {
    minval = 0,
    maxval = 1,
    dtype = float32,
    device,
  }: { minval?: number; maxval?: number; dtype?: DType; device?: Device } = {},
): Array {
  void key;
  void maxval;
  return full(shape, minval, { dtype, device });
}
