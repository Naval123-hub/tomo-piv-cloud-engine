/** Keeps route bodies flat and makes sure every rejection reaches the error
 *  middleware instead of becoming an unhandled promise rejection. */
export const route = (handler) => (req, res, next) =>
  Promise.resolve(handler(req, res, next)).catch(next);
