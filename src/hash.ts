/**
 * Polynomial hashes modulo p are good at avoiding collisions in expectation.
 * Probability-wise, it's good enough to be used for something like
 * deduplicating seen compiler expressions, although it's not adversarial.
 *
 * See https://en.wikipedia.org/wiki/Lagrange%27s_theorem_(number_theory)
 */
export class FpHash {
  value: bigint = 0n;

  #update(x: bigint) {
    // These primes were arbitrarily chosen, should be at least 10^9.
    const base = 3022769n;
    const modulus = 3189051996290219n; // Less than 2^53-1, for convenience.

    this.value = (this.value * base + x) % modulus;
  }

  update(...values: (string | boolean | bigint | null | undefined)[]): this {
    for (const x of values) {
      if (typeof x === "string") {
        for (const c of x) this.#update(BigInt(c.charCodeAt(0)));
      } else if (typeof x === "boolean") {
        this.#update(x ? 71657401n : 63640693n);
      } else if (typeof x === "bigint") {
        this.#update(x + 37832657n);
      } else if (x === null) {
        this.#update(69069841n);
      } else if (x === undefined) {
        this.#update(18145117n);
      }
    }
    return this;
  }

  static hash(
    ...values: (string | boolean | bigint | null | undefined)[]
  ): bigint {
    return new FpHash().update(...values).value;
  }
}
