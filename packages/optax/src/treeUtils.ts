import { DType, JsTree, numpy as np, tree } from "@jax-js/jax";

export function treeZerosLike(
  tr: JsTree<np.Array>,
  dtype?: DType,
): JsTree<np.Array> {
  return tree.map((x: np.Array) => np.zerosLike(x, dtype), tr);
}

export function treeOnesLike(
  tr: JsTree<np.Array>,
  dtype?: DType,
): JsTree<np.Array> {
  return tree.map((x: np.Array) => np.onesLike(x, dtype), tr);
}

function ipow(a: np.Array, order: number) {
  if (!Number.isInteger(order) || order <= 0) {
    throw new Error("Order must be a positive integer");
  }
  let result = a.ref;
  for (let i = 1; i < order; i++) {
    result = result.mul(a.ref);
  }
  a.dispose();
  return result;
}

export function treeUpdateMoment(
  updates: JsTree<np.Array>,
  moments: JsTree<np.Array>,
  decay: number,
  order: number,
): JsTree<np.Array> {
  return tree.map(
    (g: np.Array, t: np.Array) =>
      ipow(g, order)
        .mul(1 - decay)
        .add(t.mul(decay)),
    updates,
    moments,
  );
}

/** Performs bias correction, dividing by 1-decay^count. */
export function treeBiasCorrection(
  moments: JsTree<np.Array>,
  decay: number,
  count: np.Array,
): JsTree<np.Array> {
  const correction = 1 / (1 - Math.pow(decay, count.item()));
  return tree.map((t: np.Array) => t.mul(correction), moments);
}
