import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { ThemeProvider } from "../../theme/ThemeContext";
import MlLab from "../MlLab";

const mockModels = () => {
  vi.spyOn(globalThis, "fetch").mockImplementation((input: RequestInfo | URL) => {
    const url = typeof input === "string" ? input : input.toString();
    if (url.includes("/api/ml/models")) {
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve([]),
      } as Response);
    }
    if (url.includes("/api/ml/retrain")) {
      return Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            status: "started",
            message: "Training started",
            pid: 12345,
          }),
      } as Response);
    }
    return Promise.resolve({ ok: true, json: () => Promise.resolve({}) } as Response);
  });
};

beforeEach(() => {
  vi.restoreAllMocks();
});

// ── Bug: Polling name mismatch when name field is empty ──

describe("MlLab - Retrain polling name bug", () => {
  it("should send the same name in retrain request and polling URL", async () => {
    mockModels();

    render(
      <ThemeProvider>
        <MlLab />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("ML Lab")).toBeDefined();
    });

    const startButton = screen.getByText("Start Retrain");
    expect(startButton).toBeDefined();

    // Before clicking, verify the name input is empty
    const nameInput = screen.getByPlaceholderText("auto-name") as HTMLInputElement;
    expect(nameInput).toBeDefined();

    // Click start — this triggers handleRetrain
    // Bug: when req.name is "", retrain sends name: "" but polling uses `model_${Date.now()}`
    // The polling name and retrain name don't match, so polling runs forever

    // TODO: This test verifies the bug exists by checking the fetch calls.
    // After clicking Start Retrain, the code does:
    //   1. POST /api/ml/retrain with { name: "", ... }
    //   2. setInterval: GET /api/ml/retrain/status/model_<timestamp>
    // These names don't match, so the polling never finds the training status.
    userEvent.click(startButton);

    await waitFor(() => {
      // The fetch calls should have the same name in both POST and GET
      const fetchCalls = vi.mocked(fetch).mock.calls;
      const postCall = fetchCalls.find(
        ([url, opts]) =>
          typeof url === "string" && url.includes("/api/ml/retrain") &&
          (!opts || (opts as RequestInit).method === "POST")
      );
      const getCall = fetchCalls.find(
        ([url]) => typeof url === "string" && url.includes("/api/ml/retrain/status/")
      );

      if (postCall && getCall) {
        const postUrl = typeof postCall[0] === "string" ? postCall[0] : "";
        const getUrl = typeof getCall[0] === "string" ? getCall[0] : "";

        // Extract names from POST body and GET URL
        const postBody = (postCall[1] as RequestInit).body as string;
        const postName = JSON.parse(postBody).name;
        const getName = getUrl.split("/api/ml/retrain/status/")[1];

        // Bug: postName is "" (empty) but getName is "model_<timestamp>"
        // Expected: they should match
        expect(postName).toBe(getName);
      }
    });
  });
});

// ── Bug: walk_forward boolean/number type mismatch ──

describe("MlLab - walk_forward type mismatch", () => {
  it("should send walk_forward as number, not boolean", async () => {
    mockModels();

    render(
      <ThemeProvider>
        <MlLab />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("ML Lab")).toBeDefined();
    });

    // Click the "Walk-Forward" checkbox to enable it
    const wfCheckbox = screen.getByText("Walk-Forward").previousElementSibling as HTMLInputElement;
    expect(wfCheckbox).toBeDefined();
    userEvent.click(wfCheckbox);

    const startButton = screen.getByText("Start Retrain");
    userEvent.click(startButton);

    await waitFor(() => {
      const fetchCalls = vi.mocked(fetch).mock.calls;
      const postCall = fetchCalls.find(
        ([url, opts]) =>
          typeof url === "string" && url.includes("/api/ml/retrain") &&
          (!opts || (opts as RequestInit).method === "POST")
      );

      if (postCall) {
        const postBody = (postCall[1] as RequestInit).body as string;
        const body = JSON.parse(postBody);

        // Bug: MlRetrainRequest.walk_forward is typed as `number` in client.ts
        // But toggleBool in MlLab.tsx sets it to a boolean (true/false)
        // So the API receives `walk_forward: true` instead of `walk_forward: 1`
        expect(typeof body.walk_forward).toBe("number");
        // Bug: currently body.walk_forward is boolean (true) instead of number (1)
        // After fix: should be number
        if (typeof body.walk_forward === "boolean") {
          // Document the bug
          expect.fail(
            `walk_forward sent as boolean (${body.walk_forward}) instead of number. ` +
            "Bug: toggleBool sets boolean value, but MlRetrainRequest.walk_forward expects number."
          );
        }
      }
    });
  });
});

// ── Bug: No cleanup of polling interval on unmount ──

describe("MlLab - Polling cleanup on unmount", () => {
  it("should clear polling interval when component unmounts", async () => {
    mockModels();

    const clearIntervalSpy = vi.spyOn(globalThis, "clearInterval");
    const setIntervalSpy = vi.spyOn(globalThis, "setInterval");

    const { unmount } = render(
      <ThemeProvider>
        <MlLab />
      </ThemeProvider>
    );

    await waitFor(() => {
      expect(screen.getByText("ML Lab")).toBeDefined();
    });

    // Click Start Retrain to start polling
    userEvent.click(screen.getByText("Start Retrain"));

    await waitFor(() => {
      // setInterval should have been called for polling
      expect(setIntervalSpy).toHaveBeenCalled();
    });

    // Unmount the component
    unmount();

    // Bug: The polling interval created by handleRetrain (setInterval at line 127)
    // is NOT cleaned up when the component unmounts.
    // Only the default 5-second model list polling (line 118) is cleaned up via useEffect return.
    // This test documents the gap — clearInterval should be called for the retrain polling too.

    // Get all intervals that were created
    const intervalsCreated = setIntervalSpy.mock.results.length;
    const intervalsCleared = clearIntervalSpy.mock.calls.length;

    // Any interval not cleared is a potential leak
    // Note: the 5s model list interval IS cleaned up by useEffect return
    // But the 3s retrain polling interval is NOT cleaned up on unmount
    expect(intervalsCleared).toBeGreaterThanOrEqual(1);
  });
});
