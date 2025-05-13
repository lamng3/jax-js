/** @file Implementations of vjp() and partial evaluation. */

import { DType } from "../alu";
import { flatten as treeFlatten, unflatten as treeUnflatten } from "../tree";
import { invertPermutation, range, toposort, unzip2 } from "../utils";
import { pureArray, zeros } from "./array";
import {
  AbstractValue,
  add,
  bind,
  broadcast,
  flattenFun,
  fullRaise,
  mul,
  ndim,
  neg,
  newMain,
  Primitive,
  reduceSum,
  ShapedArray,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  where,
} from "./core";
import {
  abstractEvalRules,
  evalJaxpr,
  Jaxpr,
  JaxprEqn,
  Lit,
  typecheckJaxpr,
  Var,
} from "./jaxpr";
import { jvp } from "./jvp";

/** Array value that can either be known or unknown. */
class PartialVal {
  constructor(
    readonly val: Tracer | null,
    readonly aval: ShapedArray,
  ) {}

  static known(val: Tracer): PartialVal {
    return new PartialVal(val, ShapedArray.fromAval(val.aval));
  }

  static unknown(aval: AbstractValue): PartialVal {
    return new PartialVal(null, ShapedArray.fromAval(aval));
  }

  get isKnown(): boolean {
    return this.val !== null;
  }

  toString(): string {
    return this.val ? this.val.toString() : this.aval.strShort();
  }
}

function partialEvalFlat(
  f: (...args: any[]) => any,
  pvalsIn: PartialVal[],
): { jaxpr: Jaxpr; pvalsOut: PartialVal[]; consts: Tracer[] } {
  const main = newMain(PartialEvalTrace);
  const trace = new PartialEvalTrace(main);
  const tracersIn = pvalsIn.map((pval) => trace.newArg(pval));
  const outs = f(...tracersIn);
  const tracersOut: PartialEvalTracer[] = outs.map((out: TracerValue) =>
    fullRaise(trace, out),
  );
  const pvalsOut = tracersOut.map((t) => t.pval);
  const unknownTracersIn = tracersIn.filter((t) => !t.pval.isKnown);
  const unknownTracersOut = tracersOut.filter((t) => !t.pval.isKnown);
  const { jaxpr, consts } = partialEvalGraphToJaxpr(
    unknownTracersIn,
    unknownTracersOut,
  );
  return { jaxpr, pvalsOut, consts };
}

/**
 * Helper function with shared Jaxpr logic between linearize and vjp.
 *
 * Internally, vjp() looks very similar to linearize() but returns a function
 * evaluating the "transposed" linearized Jaxpr, pulling back cotangents instead
 * of pushing forward tangents.
 */
function linearizeFlatUtil(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): { primalsOut: Tracer[]; jaxpr: Jaxpr; consts: Tracer[] } {
  const pvalsIn = [
    ...primalsIn.map(PartialVal.known),
    ...primalsIn.map((t) => PartialVal.unknown(t.aval)),
  ];
  const fJvp = (...x: Tracer[]) => {
    // Args contain both primals and tangents, concatenated.
    const k = x.length / 2;
    const [primalsOut, tangentsOut] = jvp(f, x.slice(0, k), x.slice(k, 2 * k));
    return [...primalsOut, ...tangentsOut];
  };
  const { jaxpr, pvalsOut, consts } = partialEvalFlat(fJvp, pvalsIn);
  const primalPvals = pvalsOut.slice(0, pvalsOut.length / 2);
  if (!primalPvals.every((pval) => pval.isKnown)) {
    throw new TypeError(
      "Not all primal values are known after partial evaluation",
    );
  }
  const primalsOut = primalPvals.map((pval) => pval.val!);
  return { primalsOut, jaxpr, consts };
}

function linearizeFlat(
  f: (...args: any[]) => any,
  primalsIn: Tracer[],
): [Tracer[], (...args: any[]) => any] {
  const { primalsOut, jaxpr, consts } = linearizeFlatUtil(f, primalsIn);
  const fLin = (...tangents: Tracer[]) =>
    evalJaxpr(jaxpr, [...consts, ...tangents]);
  return [primalsOut, fLin];
}

export function linearize(
  f: (...primals: any[]) => any,
  ...primalsIn: any[]
): [any, (...tangents: any[]) => any] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  const [fFlat, outTree] = flattenFun(f, inTree);
  const [primalsOutFlat, fLinFlat] = linearizeFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in linearize");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const fLin = (...tangentsIn: any[]) => {
    const [tangentsInFlat, inTree2] = treeFlatten(tangentsIn);
    if (!inTree.equals(inTree2)) {
      throw new TreeMismatchError("linearize", inTree, inTree2);
    }
    const tangentsOutFlat = fLinFlat(...tangentsInFlat.map(pureArray));
    return treeUnflatten(outTree.value!, tangentsOutFlat);
  };
  return [primalsOut, fLin];
}

// Used in PartialEvalTracer to track recipes for "unknown" partial vals.
type JaxprRecipe =
  | {
      type: "LambdaBinding";
    }
  | {
      // Note: Not really a constant, actually just a "known" value translated
      // into unknown for abstract evaluation rules.
      type: "Const";
      val: Tracer;
    }
  | {
      type: "JaxprEqn";
      prim: Primitive;
      tracersIn: PartialEvalTracer[];
      params: Record<string, any>;
      avalsOut: ShapedArray[];
      tracerRefsOut: WeakRef<PartialEvalTracer>[];
    };

class PartialEvalTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly pval: PartialVal,
    readonly recipe: JaxprRecipe | null,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.pval.aval;
  }

  fullLower(): Tracer {
    if (this.pval.isKnown) {
      return this.pval.val!;
    }
    return this;
  }

  toString(): string {
    if (!this.recipe) {
      return `PartialEvalTracer(${this.pval})`;
    } else {
      return `PartialEvalTracer<${this.recipe.type}>(${this.pval})`;
    }
  }
}

class PartialEvalTrace extends Trace {
  newArg(pval: PartialVal) {
    return new PartialEvalTracer(this, pval, { type: "LambdaBinding" });
  }

  pure(val: TracerValue): Tracer {
    return new PartialEvalTracer(this, PartialVal.known(pureArray(val)), null);
  }
  lift = this.pure;

  instantiateConst(tracer: PartialEvalTracer) {
    if (!tracer.pval.isKnown) {
      return tracer;
    } else {
      // Translate known value into unknown "Const" recipe for abstract eval.
      const pval = PartialVal.unknown(ShapedArray.fromAval(tracer.aval));
      return new PartialEvalTracer(this, pval, {
        type: "Const",
        val: tracer.pval.val!,
      });
    }
  }

  processPrimitive(
    primitive: Primitive,
    tracers: PartialEvalTracer[],
    params: Record<string, any>,
  ): Tracer[] {
    if (tracers.every((t) => t.pval.isKnown)) {
      return bind(
        primitive,
        tracers.map((t) => t.fullLower()),
        params,
      );
    }
    const tracersIn = tracers.map((t) => this.instantiateConst(t));
    const avalsIn = tracersIn.map((t) => t.pval.aval);
    const avalsOut = abstractEvalRules[primitive](avalsIn, params);
    const recipe: JaxprRecipe = {
      type: "JaxprEqn",
      prim: primitive,
      tracersIn,
      params,
      avalsOut,
      tracerRefsOut: [], // Populated later on
    };
    const tracersOut = avalsOut.map(
      (aval) => new PartialEvalTracer(this, PartialVal.unknown(aval), recipe),
    );
    recipe.tracerRefsOut = tracersOut.map((t) => new WeakRef(t));
    return tracersOut;
  }
}

/**
 * Convert the graph representation of a partial eval to a standard Jaxpr.
 * Also called `tracers_to_jaxpr()` in JAX.
 */
function partialEvalGraphToJaxpr(
  tracersIn: PartialEvalTracer[],
  tracersOut: PartialEvalTracer[],
): { jaxpr: Jaxpr; consts: Tracer[] } {
  const tracerToVar = new Map<PartialEvalTracer, Var>();
  const constidToVar = new Map<Tracer, Var>();
  const constvarToVal = new Map<Var, Tracer>();
  const processedEqns = new Set<JaxprRecipe>(); // Avoid translating the same equation multiple times.
  const eqns: JaxprEqn[] = [];

  for (const t of tracersIn) {
    tracerToVar.set(t, new Var(ShapedArray.fromAval(t.aval)));
  }

  for (const t of toposort(tracersOut, (t) =>
    t.recipe?.type === "JaxprEqn" ? t.recipe.tracersIn : [],
  )) {
    if (!t.recipe) {
      throw new TypeError("Tracer is missing a recipe, cannot construct Jaxpr");
    }
    if (t.recipe.type === "LambdaBinding") {
      // Check that the binding is in the input list.
      if (!tracersIn.includes(t)) {
        throw new TypeError("LambdaBinding tracer not in input list");
      }
    } else if (t.recipe.type === "Const") {
      const val = t.recipe.val;
      let binder = constidToVar.get(val);
      if (!binder) {
        binder = new Var(ShapedArray.fromAval(val.aval));
        constidToVar.set(val, binder);
        constvarToVal.set(binder, val);
      }
      tracerToVar.set(t, binder);
    } else if (t.recipe.type === "JaxprEqn") {
      if (!processedEqns.has(t.recipe)) {
        processedEqns.add(t.recipe);
        const tracersIn = t.recipe.tracersIn.map((t) => tracerToVar.get(t)!);
        const outBinders = t.recipe.avalsOut.map((aval) => new Var(aval));
        for (let i = 0; i < outBinders.length; i++) {
          const tracerOut = t.recipe.tracerRefsOut[i].deref();
          if (tracerOut) {
            tracerToVar.set(tracerOut, outBinders[i]);
          }
        }
        eqns.push(
          new JaxprEqn(t.recipe.prim, tracersIn, t.recipe.params, outBinders),
        );
      }
    }
  }

  const [constvars, consts] = unzip2(constvarToVal.entries());
  const inBinders = [
    ...constvars,
    ...tracersIn.map((t) => tracerToVar.get(t)!),
  ];
  const outVars = tracersOut.map((t) => tracerToVar.get(t)!);
  const jaxpr = new Jaxpr(inBinders, eqns, outVars);
  typecheckJaxpr(jaxpr); // sanity check
  return { jaxpr: jaxpr.simplify(), consts };
}

// implementation of vjp and grad

/** Marker type for pullback, used by transpose rules. */
class UndefPrimal {
  readonly aval: ShapedArray;

  constructor(aval: AbstractValue) {
    this.aval = ShapedArray.fromAval(aval);
  }
}

/**
 * Evaluate the backward pass over a linearized Jaxpr (pullback of cotangents).
 *
 * Will raise a TypeError if the provided Jaxpr is not a linear function of its,
 * inputs, as general expressions cannot be transposed.
 */
function evalJaxprTransposed(
  jaxpr: Jaxpr,
  args: (Tracer | UndefPrimal)[],
  cotangents: Tracer[],
): Tracer[] {
  const knownPrimals = new Map<Var, Tracer>();
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (!(args[i] instanceof UndefPrimal)) {
      knownPrimals.set(jaxpr.inBinders[i], args[i] as Tracer);
    }
  }

  const ctStore = new Map<Var, Tracer>();

  const readCotangent = (v: Var) => {
    const ct = ctStore.get(v);
    if (ct) {
      // We should read a cotangent at most once, as an out binder.
      ctStore.delete(v);
      return ct;
    } else {
      return zeros(v.aval.shape, { dtype: v.aval.dtype });
    }
  };

  const writeCotangent = (v: Var, ct: Tracer | null) => {
    if (ct !== null) {
      const oldCt = ctStore.get(v);
      // May need to accumulate cotangents if used in multiple JaxprEqns.
      if (oldCt) ctStore.set(v, add(oldCt, ct));
      else ctStore.set(v, ct);
    }
  };

  for (let i = 0; i < jaxpr.outs.length; i++) {
    const v = jaxpr.outs[i];
    if (v instanceof Var) writeCotangent(v, cotangents[i]);
  }

  for (let i = jaxpr.eqns.length - 1; i >= 0; i--) {
    const eqn = jaxpr.eqns[i];
    // Inputs are primalsIn and cotangentsOut, outputs are cotangentsIn. We're
    // using the known primal values to _pull back_  cotangents for unknown
    // values. Tricky!
    const primalsIn = eqn.inputs.map((v) =>
      v instanceof Lit
        ? v.val
        : (knownPrimals.get(v) ?? new UndefPrimal(v.aval)),
    );
    const cotangentsOut = eqn.outBinders.map(readCotangent);
    const rule = transposeRules[eqn.primitive];
    if (!rule) {
      throw new TypeError(`Backward pass not implemented for ${eqn.primitive}`);
    }
    const cotangentsIn = rule(cotangentsOut, primalsIn, eqn.params);
    for (let j = 0; j < jaxpr.inBinders.length; j++) {
      const v = eqn.inputs[j];
      if (v instanceof Var && !knownPrimals.has(v)) {
        writeCotangent(v, cotangentsIn[j]);
      }
    }
  }

  const results: Tracer[] = [];
  for (let i = 0; i < jaxpr.inBinders.length; i++) {
    if (args[i] instanceof UndefPrimal) {
      results.push(readCotangent(jaxpr.inBinders[i]));
    }
  }
  return results;
}

class NonlinearError extends TypeError {
  constructor(primitive: Primitive) {
    super(`Nonlinear operation in backward pass for ${primitive}`);
  }
}

type TransposeRule = (
  cotangents: Tracer[],
  primals: (Tracer | UndefPrimal)[],
  params: any,
) => (Tracer | null)[];

// You need a transpose rule for a primitive p if:
//  - p is used in jvpRules, while computing a tangent (not primal)
//  - in this use, at least one argument to p is a tangent
//
// This computes a backward pass, so it pulls back cotangents to the inputs of p
// that are UndefPrimal (i.e., tangents that weren't sent forward).
const transposeRules: Partial<Record<Primitive, TransposeRule>> = {
  [Primitive.Mul]([ct], [x, y]) {
    // BUG: Doesn't handle broadcasting.
    if (x instanceof UndefPrimal === y instanceof UndefPrimal)
      throw new NonlinearError(Primitive.Mul);
    return x instanceof UndefPrimal
      ? [mul(ct, y as Tracer), null]
      : [null, mul(x as Tracer, ct)];
  },
  [Primitive.Neg]([ct], [x]) {
    if (!(x instanceof UndefPrimal)) throw new NonlinearError(Primitive.Neg);
    return [neg(ct)];
  },
  [Primitive.Add]([ct], [x, y]) {
    // BUG: Doesn't handle broadcasting.
    if (!(x instanceof UndefPrimal || y instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Add);
    return [ct, ct];
  },
  [Primitive.ReduceSum]([ct], [x], { axis }: { axis: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.ReduceSum);
    return [broadcast(ct, x.aval.shape, axis)];
  },
  // BUG: Doesn't handle broadcasting.
  [Primitive.Where]([ct], [cond, x, y]) {
    // Cotangent should be zero be zero where cond doesn't apply.
    const cts: (Tracer | null)[] = [null, null, null];
    if (cond instanceof UndefPrimal) throw new NonlinearError(Primitive.Where);
    if (x instanceof UndefPrimal) {
      cts[1] = where(cond, ct, zeros(x.aval.shape, { dtype: x.aval.dtype }));
    }
    if (y instanceof UndefPrimal) {
      cts[2] = where(cond, zeros(y.aval.shape, { dtype: y.aval.dtype }), ct);
    }
    return cts;
  },
  [Primitive.Transpose]([ct], [x], { perm }: { perm?: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Transpose);
    const tperm = perm ? invertPermutation(perm) : range(x.aval.ndim).reverse();
    return [transpose(ct, tperm)];
  },
  [Primitive.Broadcast]([ct], [x], { axis }: { axis: number[] }) {
    if (!(x instanceof UndefPrimal))
      throw new NonlinearError(Primitive.Broadcast);
    return [reduceSum(ct, axis)];
  },
};

function vjpFlat(
  f: (...x: Tracer[]) => Tracer[],
  primalsIn: Tracer[],
): [Tracer[], (...cotangents: Tracer[]) => Tracer[]] {
  const { primalsOut, jaxpr, consts } = linearizeFlatUtil(f, primalsIn);
  const transposeInputs = [
    ...consts,
    // Explcitly list which arguments should be transposed.
    ...primalsIn.map((t) => new UndefPrimal(t.aval)),
  ];
  // Pullback cotangents to the UndefPrimal transpose inputs.
  const fVjp = (...cotangents: Tracer[]) =>
    evalJaxprTransposed(jaxpr, transposeInputs, cotangents);
  return [primalsOut, fVjp];
}

export function vjp(
  f: (...primals: any) => any,
  ...primalsIn: any
): [any, (...cotangents: any) => any] {
  const [primalsInFlat, inTree] = treeFlatten(primalsIn);
  const [fFlat, outTree] = flattenFun(f, inTree);
  const [primalsOutFlat, fVjpFlat] = vjpFlat(
    fFlat,
    primalsInFlat.map(pureArray),
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in vjp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);

  // "cotangentsOut" because pullback
  const fVjp = (cotangentsOut: any) => {
    const [cotangentsOutFlat, outTree2] = treeFlatten(cotangentsOut);
    if (!outTree.value!.equals(outTree2)) {
      throw new TreeMismatchError("vjp", outTree.value!, outTree2);
    }
    const cotangentsInFlat = fVjpFlat(...cotangentsOutFlat.map(pureArray));
    return treeUnflatten(inTree, cotangentsInFlat);
  };

  return [primalsOut, fVjp];
}

export function grad(f: (...primals: any) => Tracer) {
  return (...x: any) => {
    const [y, fVjp] = vjp(f, ...x);
    if (!(y instanceof Tracer) || ndim(y) !== 0) {
      throw new TypeError("grad requires a scalar output");
    }
    if (y.dtype !== DType.Float32) {
      throw new TypeError("grad currently only supports float32");
    }
    // JAX convention, differentiate with respect to the first argument.
    return fVjp(pureArray(1))[0];
  };
}
