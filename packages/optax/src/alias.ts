import { GradientTransformation, ScalarOrSchedule } from "./base";
import { chain } from "./combine";
import {
  scaleByAdam,
  ScaleByAdamOptions,
  scaleByLearningRate,
} from "./transform";

/** Stochastic gradient descent. */
export function sgd(learningRate: ScalarOrSchedule): GradientTransformation {
  // TODO: Add momentum and nesterov options (via optax.trace).
  return scaleByLearningRate(learningRate);
}

/** The Adam optimizer. */
export function adam(
  learningRate: ScalarOrSchedule,
  opts: ScaleByAdamOptions,
): GradientTransformation {
  return chain(scaleByAdam(opts), scaleByLearningRate(learningRate));
}
