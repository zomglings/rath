/**
 * resolveModel finds z.ai's glm-5.3 in the bundled pi-ai registry. No API call.
 */
import assert from "node:assert/strict";
import { resolveModel } from "../models.js";

const model = resolveModel("zai/glm-5.3");
assert.equal(model.provider, "zai");
assert.equal(model.id, "glm-5.3");
console.log(`zai/glm-5.3 -> ${model.api} ${model.baseUrl}`);
