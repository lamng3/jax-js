import * as tf from "@tensorflow/tfjs-core";
import "@tensorflow/tfjs-core/dist/public/chained_ops/register_all_chained_ops";
import "@tensorflow/tfjs-core/dist/register_all_gradients";
import "@tensorflow/tfjs-backend-cpu";

export enum DType {
  Float32 = "float32",
  Int32 = "int32",
  Bool = "bool",
  Complex64 = "complex64",
}

export enum Primitive {
  Add = "add",
  Mul = "mul",
  Neg = "neg",
  Sin = "sin",
  Cos = "cos",
  ReduceSum = "reduce_sum",
  Greater = "greater",
  Less = "less",
  Transpose = "transpose",
  Broadcast = "broadcast",
}

export function add(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Add, [x, y]);
}

export function mul(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Mul, [x, y]);
}

export function neg(x: TracerValue) {
  return bind1(Primitive.Neg, [x]);
}

export function sin(x: TracerValue) {
  return bind1(Primitive.Sin, [x]);
}

export function cos(x: TracerValue) {
  return bind1(Primitive.Cos, [x]);
}

export function greater(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Greater, [x, y]);
}

export function less(x: TracerValue, y: TracerValue) {
  return bind1(Primitive.Less, [x, y]);
}

export function transpose(x: TracerValue, perm?: number[]) {
  return bind1(Primitive.Transpose, [x], { perm });
}

export function broadcast(x: TracerValue, shape: number[], axes: number[]) {
  return bind1(Primitive.Broadcast, [x], { shape, axes });
}

export function reduceSum(x: TracerValue, axis?: number | number[]) {
  if (axis === null) {
    if (x instanceof Tracer) {
      axis = [...JsArray(x.aval.shape.length).keys()];
    } else {
      axis = [];
    }
  }
  if (typeof axis === "number") {
    axis = [axis];
  }
  return bind1(Primitive.ReduceSum, [x], { axis });
}

function bind1(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {}
) {
  const [results] = bind(prim, args, params);
  return results;
}

type MainTrace = {
  level: number;
  traceType: new (main: MainTrace) => Trace; // Concrete Trace subclass.
  globalData: any | null;
};

let traceStack: MainTrace[] = [];
let dynamicTrace: MainTrace | null = null;

// Push an interpreter onto the trace stack. Use this like:
// `using { main } = newMain(...);`
function newMain(
  traceType: any,
  globalData: any | null = null
): Disposable & MainTrace {
  const level = traceStack.length;
  const main = { level, traceType, globalData };
  traceStack.push(main);
  return Object.assign(main, {
    [Symbol.dispose]() {
      traceStack.pop();
    },
  });
}

type TracerValue = Tracer | number | boolean;

abstract class Trace {
  constructor(public main: MainTrace) {}

  abstract pure(val: TracerValue): Tracer;
  abstract lift(val: Tracer): Tracer;

  abstract processPrimitive(
    primitive: Primitive,
    tracers: Tracer[],
    params: Record<string, any>
  ): Tracer[];
}

interface AbstractValue {
  shape: number[];
  dtype: DType;

  _neg: (x: Tracer) => Tracer;
  _add: (x: Tracer, y: Tracer) => Tracer;
  _mul: (x: Tracer, y: Tracer) => Tracer;
  _gt: (x: Tracer, y: Tracer) => Tracer;
  _lt: (x: Tracer, y: Tracer) => Tracer;
}

abstract class Tracer {
  readonly _trace: Trace;

  constructor(trace: Trace) {
    this._trace = trace;
  }

  abstract get aval(): AbstractValue;

  fullLower() {
    return this; // default implementation
  }

  neg() {
    return this.aval._neg(this);
  }
  add(other: Tracer) {
    return this.aval._add(this, other);
  }
  mul(other: Tracer) {
    return this.aval._mul(this, other);
  }
  gt(other: Tracer) {
    return this.aval._gt(this, other);
  }
  lt(other: Tracer) {
    return this.aval._lt(this, other);
  }

  // TODO
  /*

  def __getattr__(self, name):
    try:
      return getattr(self.aval, name)
    except AttributeError:
      raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")
*/
}

const JsArray = globalThis.Array;

class ShapedArray implements AbstractValue {
  readonly arrayAbstractionLevel: number = 1;

  constructor(
    public readonly shape: number[],
    public readonly dtype: DType
  ) {}

  get ndim() {
    return this.shape.length;
  }

  // See note about primitive wrappers with fudged types.
  _neg = neg as any;
  _add = add as any;
  _mul = mul as any;
  _gt = greater as any;
  _lt = less as any;

  strShort() {
    return `${this.dtype}[${this.shape.join(",")}]`;
  }

  equals(other: ShapedArray) {
    return (
      this === other ||
      (this.constructor === other.constructor &&
        this.shape.length === other.shape.length &&
        this.shape.every((d, i) => d === other.shape[i]))
    );
  }
}

class ConcreteArray extends ShapedArray {
  readonly arrayAbstractionLevel: number = 2;

  constructor(public readonly val: tf.Tensor) {
    super(val.shape, val.dtype as any);
  }
}

/**
 * Equivalent to `jnp.Array` from JAX, a tensor type.
 *
 * Not to be confused with the JavaScript "Array" constructor. Avoid importing this into your code's
 * namespace if you're already using the JavaScript "Array" type by name.
 */
export class Array extends Tracer {
  readonly dtype: DType;

  constructor(public readonly data: tf.Tensor) {
    super(baseArrayTrace);
    if (Object.values(DType).includes(data.dtype as any)) {
      this.dtype = data.dtype as DType;
    } else {
      throw new TypeError(`Unsupported dtype: ${data.dtype}`);
    }
  }

  get aval(): AbstractValue {
    return new ConcreteArray(this.data);
  }

  /** Convert this array into a JavaScript object (blocking). */
  js() {
    return this.data.arraySync();
  }

  /** Convert this array into a JavaScript object, asynchronously. */
  async jsAsync() {
    return await this.data.array();
  }
}

/** If x is a value, lift it into an array, otherwise leave it be. */
function pureArray(x: TracerValue): Tracer {
  if (x instanceof Tracer) {
    return x;
  } else {
    return new Array(tf.scalar(x));
  }
}

function getAval(x: TracerValue): AbstractValue {
  if (x instanceof Tracer) {
    return x.aval;
  } else if (typeof x === "boolean" || typeof x === "number") {
    return new ConcreteArray(tf.scalar(x));
  } else {
    throw new TypeError(`Unknown value: ${x}`);
  }
}

function bind(
  prim: Primitive,
  args: TracerValue[],
  params: Record<string, any> = {}
) {
  const topTrace = findTopTrace(args);
  const tracers = args.map((arg) => fullRaise(topTrace, arg));
  const outs = topTrace.processPrimitive(prim, tracers, params);
  return outs.map((out) => out.fullLower());
}

function findTopTrace(xs: TracerValue[]): Trace {
  let topMain: MainTrace = traceStack[0];
  for (const x of xs) {
    if (x instanceof Tracer && x._trace.main.level > topMain.level) {
      topMain = x._trace.main;
    }
  }
  if (dynamicTrace && dynamicTrace.level > topMain.level) {
    topMain = dynamicTrace;
  }
  return new topMain.traceType(topMain);
}

function fullRaise(trace: Trace, val: TracerValue): Tracer {
  if (!(val instanceof Tracer)) {
    // remember to assert type(val) in jax_types
    return trace.pure(val);
  }
  const level = trace.main.level;
  if (Object.is(val._trace.main, trace.main)) {
    return val;
  } else if (val._trace.main.level < level) {
    return trace.lift(val);
  } else if (val._trace.main.level > level) {
    throw new Error(
      `Can't lift Tracer level ${val._trace.main.level} to level ${level}`
    );
  } else {
    throw new Error(`Different traces at same level: ${val._trace}, ${trace}.`);
  }
}

class EvalTrace extends Trace {
  // No boxing in Tracers needed.
  pure = (x: TracerValue) => pureArray(x);
  lift = (x: Tracer) => x;

  processPrimitive(
    primitive: Primitive,
    tracers: Tracer[],
    params: Record<string, any>
  ): Tracer[] {
    return implRules[primitive](tracers as Array[], params);
  }
}

// Special bottom of the stack.
traceStack.push({ level: 0, traceType: EvalTrace, globalData: null });
const baseArrayTrace = new EvalTrace(traceStack[0]);

type ImplRule = (tracers: Array[], params: Record<string, any>) => Array[];

const implRules: Record<Primitive, ImplRule> = {
  [Primitive.Add]([x, y]) {
    return [new Array(tf.add(x.data, y.data))];
  },
  [Primitive.Mul]([x, y]) {
    return [new Array(tf.mul(x.data, y.data))];
  },
  [Primitive.Neg]([x]) {
    return [new Array(tf.neg(x.data))];
  },
  [Primitive.Sin]([x]) {
    return [new Array(tf.sin(x.data))];
  },
  [Primitive.Cos]([x]) {
    return [new Array(tf.cos(x.data))];
  },
  [Primitive.ReduceSum]([x], { axis }: { axis?: number | number[] }) {
    return [new Array(tf.sum(x.data, axis))];
  },
  [Primitive.Greater]([x, y]) {
    return [new Array(tf.greater(x.data, y.data))];
  },
  [Primitive.Less]([x, y]) {
    return [new Array(tf.less(x.data, y.data))];
  },
  [Primitive.Transpose]([x], { perm }: { perm?: number[] }) {
    return [new Array(tf.transpose(x.data, perm))];
  },
  [Primitive.Broadcast](
    [x],
    { shape, axes }: { shape?: number[]; axes?: number[] }
  ) {
    if (shape === undefined || axes === undefined) {
      throw new Error("Must provide shape and axes to broadcast");
    }
    let data = x.data;
    for (const axis of axes.toSorted()) {
      data = tf.expandDims(data, axis);
    }
    return [new Array(tf.broadcastTo(data, shape))];
  },
};

function zerosLike(val: TracerValue): Array {
  const aval = getAval(val);
  return new Array(tf.zeros(aval.shape, aval.dtype));
}

function unzip2<T, U>(pairs: [T, U][]): [T[], U[]] {
  const lst1: T[] = [];
  const lst2: U[] = [];
  for (const [x, y] of pairs) {
    lst1.push(x);
    lst2.push(y);
  }
  return [lst1, lst2];
}

function zip<T, U>(xs: T[], ys: U[]): [T, U][] {
  return xs.map((x, i) => [x, ys[i]]);
}

class JVPTracer extends Tracer {
  constructor(
    trace: Trace,
    public readonly primal: Tracer,
    public readonly tangent: Tracer
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.primal.aval;
  }
}

class JVPTrace extends Trace {
  pure(val: TracerValue) {
    return this.lift(pureArray(val));
  }

  lift(val: Tracer): Tracer {
    return new JVPTracer(this, val, zerosLike(val));
  }

  processPrimitive(
    primitive: Primitive,
    tracers: JVPTracer[],
    params: Record<string, any>
  ): JVPTracer[] {
    const [primalsIn, tangentsIn] = unzip2(
      tracers.map((x) => [x.primal, x.tangent])
    );
    const jvpRule: JvpRule | undefined = jvpRules[primitive];
    if (jvpRule === undefined) {
      throw new Error(`No JVP rule for: ${primitive}`);
    }
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(
      ([x, t]) => new JVPTracer(this, x, t)
    );
  }
}

type JvpRule = (
  primal: Tracer[],
  tangents: Tracer[],
  params: Record<string, any>
) => [Tracer[], Tracer[]];

const jvpRules: Partial<Record<Primitive, JvpRule>> = {
  [Primitive.Add]([x, y], [dx, dy]) {
    return [[x.add(y)], [dx.add(dy)]];
  },
  [Primitive.Mul]([x, y], [dx, dy]) {
    return [[x.mul(y)], [x.mul(dy).add(dx.mul(y))]];
  },
  [Primitive.Neg]([x], [dx]) {
    return [[x.neg()], [dx.neg()]];
  },
  [Primitive.Sin]([x], [dx]) {
    return [[sin(x)], [cos(x).mul(dx)]];
  },
  [Primitive.Cos]([x], [dx]) {
    return [[cos(x)], [neg(sin(x)).mul(dx)]];
  },
  [Primitive.ReduceSum]([x], [dx], { axis }: { axis?: number | number[] }) {
    return [[reduceSum(x, axis)], [reduceSum(dx, axis)]];
  },
  [Primitive.Greater]([x, y], _tangents) {
    const outPrimal = greater(x, y);
    return [[outPrimal], [zerosLike(outPrimal)]];
  },
  [Primitive.Less]([x, y], _tangents) {
    const outPrimal = less(x, y);
    return [[outPrimal], [zerosLike(outPrimal)]];
  },
};

export function jvpV1(
  f: (...x: TracerValue[]) => Tracer,
  primals: TracerValue[],
  tangents: TracerValue[]
): [Tracer, Tracer] {
  using main = newMain(JVPTrace);
  const trace = new JVPTrace(main);
  const tracersIn = zip(primals, tangents).map(
    ([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t))
  );
  const out = f(...tracersIn);
  const tracerOut = fullRaise(trace, out) as JVPTracer;
  return [tracerOut.primal, tracerOut.tangent];
}

export function deriv(
  f: (x: TracerValue) => Tracer
): (x: TracerValue) => Tracer {
  return (x) => {
    const [y, dy] = jvpV1(f, [x], [1.0]);
    return dy;
  };
}
