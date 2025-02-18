import { expect, suite, test } from "vitest";
import { numpy as np, makeJaxpr, jvp } from "jax-js";

suite("jax.makeJaxpr()", () => {
  test("tracks a nullary function", () => {
    const { jaxpr, consts } = makeJaxpr(() => np.mul(2, 2))();
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda  .
        let v_3:float32[] = mul 2 2
        in ( v_3 ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("tracks a unary function", () => {
    const { jaxpr, consts } = makeJaxpr((x: np.Array) => np.mul(x.add(2), x))(
      np.array([
        [2, 4, 10],
        [1, 1, 1],
      ])
    );
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda v_1:float32[2,3] .
        let v_3:float32[2,3] = add v_1 2
            v_4:float32[2,3] = mul v_3 v_1
        in ( v_4 ) }"
    `);
    expect(consts).toEqual([]);
  });

  test("composes with jvp", () => {
    const f = (x: np.Array) => np.mul(x.add(2), x);
    const fdot = (x: np.Array) => jvp(f, [x], [np.array(1)])[1];

    const { jaxpr, consts } = makeJaxpr(fdot)(np.array(2));
    expect(jaxpr.toString()).toMatchInlineSnapshot(`
      "{ lambda v_1:float32[] .
        let v_3:float32[] = add v_1 2
            v_6:float32[] = add 1 0
            v_7:float32[] = mul v_3 v_1
            v_8:float32[] = mul v_3 1
            v_9:float32[] = mul v_6 v_1
            v_10:float32[] = add v_8 v_9
        in ( v_10 ) }"
    `);
    expect(consts).toEqual([]);
  });
});
