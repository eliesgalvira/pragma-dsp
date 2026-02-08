/**
 * A Deferred wraps a Promise and exposes its resolve/reject handles,
 * letting you settle it from the outside at any point in time.
 */
export class Deferred<T, E = unknown> {
  promise: Promise<T>;
  resolve: (value: T | PromiseLike<T>) => void = () => null;
  reject: (reason?: E) => void = () => null;

  constructor() {
    this.promise = new Promise((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }
}
