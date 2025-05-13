import { flatten as treeFlatten, unflatten as treeUnflatten } from "../tree";
import { unzip2, zip } from "../utils";
import { pureArray, zerosLike } from "./array";
import {
  AbstractValue,
  broadcast,
  compare,
  CompareOp,
  cos,
  flattenFun,
  fullRaise,
  neg,
  newMain,
  Primitive,
  reduceSum,
  sin,
  Trace,
  Tracer,
  TracerValue,
  transpose,
  TreeMismatchError,
  where,
} from "./core";

class JVPTracer extends Tracer {
  constructor(
    trace: Trace,
    readonly primal: Tracer,
    readonly tangent: Tracer,
  ) {
    super(trace);
  }

  get aval(): AbstractValue {
    return this.primal.aval;
  }

  toString(): string {
    return `JVPTracer(${this.primal}, ${this.tangent})`;
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
    params: Record<string, any>,
  ): JVPTracer[] {
    const [primalsIn, tangentsIn] = unzip2(
      tracers.map((x) => [x.primal, x.tangent]),
    );
    const jvpRule: JvpRule | undefined = jvpRules[primitive];
    if (jvpRule === undefined) {
      throw new Error(`No JVP rule for: ${primitive}`);
    }
    const [primalsOut, tangentsOut] = jvpRule(primalsIn, tangentsIn, params);
    return zip(primalsOut, tangentsOut).map(
      ([x, t]) => new JVPTracer(this, x, t),
    );
  }
}

type JvpRule = (
  primals: Tracer[],
  tangents: Tracer[],
  params: any,
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
  [Primitive.ReduceSum]([x], [dx], { axis }: { axis: number[] }) {
    return [[reduceSum(x, axis)], [reduceSum(dx, axis)]];
  },
  [Primitive.Compare]([x, y], _tangents, { op }: { op: CompareOp }) {
    const primal = compare(x, y, op);
    return [[primal], [zerosLike(primal)]];
  },
  [Primitive.Where]([cond, x, y], [_, dx, dy]) {
    return [[where(cond, x, y)], [where(cond, dx, dy)]];
  },
  [Primitive.Transpose]([x], [dx], { perm }: { perm?: number[] }) {
    return [[transpose(x, perm)], [transpose(dx, perm)]];
  },
  [Primitive.Broadcast](
    [x],
    [dx],
    { shape, axis }: { shape: number[]; axis: number[] },
  ) {
    return [[broadcast(x, shape, axis)], [broadcast(dx, shape, axis)]];
  },
};

function jvpFlat(
  f: (...x: TracerValue[]) => TracerValue[],
  primals: TracerValue[],
  tangents: TracerValue[],
): [Tracer[], Tracer[]] {
  using main = newMain(JVPTrace);
  const trace = new JVPTrace(main);
  const tracersIn = zip(primals, tangents).map(
    ([x, t]) => new JVPTracer(trace, pureArray(x), pureArray(t)),
  );
  const outs = f(...tracersIn);
  const tracersOut = outs.map((out) => fullRaise(trace, out) as JVPTracer);
  return unzip2(tracersOut.map((t) => [t.primal, t.tangent]));
}

export function jvp(
  f: (...x: any[]) => any,
  primals: any[],
  tangents: any[],
): [any, any] {
  const [primalsFlat, inTree] = treeFlatten(primals);
  const [tangentsFlat, inTree2] = treeFlatten(tangents);
  if (!inTree.equals(inTree2)) {
    throw new TreeMismatchError("jvp", inTree, inTree2);
  }

  const [flatFun, outTree] = flattenFun(f, inTree);

  const [primalsOutFlat, tangentsOutFlat] = jvpFlat(
    flatFun,
    primalsFlat,
    tangentsFlat,
  );
  if (outTree.value === undefined) {
    throw new Error("outTree was not set in jvp");
  }
  const primalsOut = treeUnflatten(outTree.value, primalsOutFlat);
  const tangentsOut = treeUnflatten(outTree.value, tangentsOutFlat);
  return [primalsOut, tangentsOut];
}
