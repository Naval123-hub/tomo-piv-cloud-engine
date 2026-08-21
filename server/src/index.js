/**
 * Tomo-PIV API Gateway -- entry point.
 *
 *   npm run dev     (node --watch)
 *   npm start
 *
 * Sits between the React client and the Python GPU backend (app_modal.py),
 * doing the three things a browser cannot do safely:
 *
 *   1. Signs Cloudflare R2 URLs, so raw frames upload and results download
 *      directly to/from object storage -- multi-GB payloads never pass through
 *      this process or a GPU container.
 *   2. Reads job status straight out of Redis, which the Python worker already
 *      writes to. Polling a running job therefore costs no container time;
 *      Modal is contacted only to start or cancel work.
 *   3. Translates friendly parameter names into the pipeline's constants and
 *      validates ranges before a job is allowed to occupy a T4.
 */
import { createApp } from "./app.js";
import { MODAL_API_URL, PORT, R2, REDIS_URL, r2Configured } from "./config.js";

const app = createApp();

app.listen(PORT, () => {
  console.log(`\n  Tomo-PIV API gateway  ->  http://localhost:${PORT}`);
  console.log(`    R2     : ${r2Configured ? `bucket ${R2.bucket}` : "NOT CONFIGURED"}`);
  console.log(`    Redis  : ${REDIS_URL ? "configured" : "NOT CONFIGURED (status proxied from Modal)"}`);
  console.log(`    Modal  : ${MODAL_API_URL || "NOT CONFIGURED"}\n`);
});

export default app;
