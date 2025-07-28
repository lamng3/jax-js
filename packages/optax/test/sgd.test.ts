import { grad, numpy as np } from "@jax-js/jax";
import { applyUpdates, sgd, squaredError } from "@jax-js/optax";
import { expect, test } from "vitest";

test("stochastic gradient descent", () => {
  let params = np.array([1.0, 2.0, 3.0]);

  const solver = sgd(0.11);
  let optState = solver.init(params.ref);
  let updates: np.Array;

  const f = (x: np.Array) => squaredError(x, np.ones([3])).sum();
  const paramsGrad = grad(f)(params.ref);
  [updates, optState] = solver.update(paramsGrad, optState);
  params = applyUpdates(params, updates);

  expect(params).toBeAllclose([1.0, 1.78, 2.56]);
});
