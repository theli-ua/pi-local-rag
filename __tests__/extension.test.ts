import { vi, describe, it, expect } from "vitest";

vi.mock("@xenova/transformers", () => ({
  pipeline: vi.fn().mockResolvedValue(
    vi.fn().mockImplementation(async () => ({
      data: new Float32Array(384).fill(0.1),
    }))
  ),
}));

import registerExtension from "../index.ts";

function makeMockApi() {
  return {
    registerCommand: vi.fn(),
    registerTool: vi.fn(),
    on: vi.fn(),
  };
}

describe("extension registration", () => {
  it("registers the rag command", () => {
    const api = makeMockApi();
    registerExtension(api as any);
    expect(api.registerCommand).toHaveBeenCalledWith("rag", expect.objectContaining({ handler: expect.any(Function) }));
  });

  it("registers exactly three tools", () => {
    const api = makeMockApi();
    registerExtension(api as any);
    expect(api.registerTool).toHaveBeenCalledTimes(3);
  });

  it("registers rag_index, rag_query, and rag_status tools", () => {
    const api = makeMockApi();
    registerExtension(api as any);
    const names = api.registerTool.mock.calls.map((call: any[]) => call[0].name);
    expect(names).toContain("rag_index");
    expect(names).toContain("rag_query");
    expect(names).toContain("rag_status");
  });

  it("registers the before_agent_start hook", () => {
    const api = makeMockApi();
    registerExtension(api as any);
    expect(api.on).toHaveBeenCalledWith("before_agent_start", expect.any(Function));
  });

  it("all registered tools have execute functions", () => {
    const api = makeMockApi();
    registerExtension(api as any);
    for (const [tool] of api.registerTool.mock.calls) {
      expect(typeof tool.execute).toBe("function");
    }
  });

  it("all registered tools have non-empty descriptions", () => {
    const api = makeMockApi();
    registerExtension(api as any);
    for (const [tool] of api.registerTool.mock.calls) {
      expect(tool.description.length).toBeGreaterThan(0);
    }
  });
});
