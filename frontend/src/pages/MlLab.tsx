import { useEffect, useState, useCallback } from "react";
import { api, type MlModelInfo, type MlRetrainRequest } from "../api/client";
import { useTheme } from "../theme/ThemeContext";

const INITIAL_REQ: MlRetrainRequest = {
  symbols: "NVDA,AMD,VOO,SPY,META",
  years: 20,
  name: "",
  model_types: "rf,gbt",
  beat_baselines: false,
  grid_search: false,
  walk_forward: 0,
  stacking: false,
  meta_labeling: false,
  regularize: false,
  prune: 0,
  kelly: false,
  auto_threshold: false,
  labeling: "next_bar",
  forecast_horizon: 1,
  regime_aware: false,
};

function MetricsCard({ label, value, color }: { label: string; value: string | number; color?: string }) {
  const { colors } = useTheme();
  return (
    <div style={{ textAlign: "center", padding: "8px 12px", background: colors.surface, borderRadius: 6, minWidth: 80 }}>
      <div style={{ fontSize: 11, color: colors.text, opacity: 0.7 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color: color || colors.text }}>{value}</div>
    </div>
  );
}

function ModelCard({ model, onDelete }: { model: MlModelInfo; onDelete: () => void }) {
  const { colors } = useTheme();
  const m = model.validation_metrics;
  const ret = m?.total_return_pct ?? 0;
  const sharpe = m?.sharpe_ratio ?? 0;
  const dd = m?.max_drawdown_pct ?? 0;
  const wr = m?.win_rate_pct ?? 0;
  const trades = m?.num_trades ?? 0;

  return (
    <div
      style={{
        background: colors.surface,
        border: `1px solid ${colors.border}`,
        borderRadius: 8,
        padding: 16,
        marginBottom: 12,
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 8 }}>
        <div>
          <strong style={{ fontSize: 15 }}>{model.name}</strong>
          <span style={{ marginLeft: 8, fontSize: 12, color: colors.text, opacity: 0.6 }}>
            v{model.version} &middot; {model.model_type}
          </span>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <span style={{ fontSize: 11, color: colors.text, opacity: 0.6 }}>
            {model.train_date?.slice(0, 10)}
          </span>
          <span style={{ fontSize: 11, color: model.beat_baselines ? "green" : colors.text, opacity: 0.7 }}>
            {model.beat_baselines ? "🏆" : ""}
          </span>
          <button
            onClick={onDelete}
            style={{
              padding: "3px 10px",
              background: colors.surface,
              color: "red",
              border: `1px solid ${colors.border}`,
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 12,
            }}
          >
            Delete
          </button>
        </div>
      </div>
      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
        <MetricsCard label="Return" value={`${ret.toFixed(1)}%`} color={ret >= 0 ? "green" : "red"} />
        <MetricsCard label="Sharpe" value={sharpe.toFixed(2)} />
        <MetricsCard label="Max DD" value={`${dd.toFixed(1)}%`} color="red" />
        <MetricsCard label="Win Rate" value={`${wr.toFixed(0)}%`} />
        <MetricsCard label="Trades" value={trades} />
      </div>
      <div style={{ marginTop: 8, fontSize: 11, color: colors.text, opacity: 0.6 }}>
        {model.train_symbols.join(", ")}
        {model.context_symbols?.length ? ` | ctx: ${model.context_symbols.join(", ")}` : ""}
        &nbsp;| versions: {model.versions.join(", ")}
      </div>
    </div>
  );
}

export default function MlLab() {
  const { colors } = useTheme();
  const [models, setModels] = useState<MlModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [req, setReq] = useState<MlRetrainRequest>({ ...INITIAL_REQ });
  const [trainingStatus, setTrainingStatus] = useState<string | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      setModels(await api.getMlModels());
    } catch {
      /* server not running */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
    const interval = setInterval(fetchModels, 5000);
    return () => clearInterval(interval);
  }, [fetchModels]);

  const handleRetrain = async () => {
    setTrainingStatus("Starting training...");
    try {
      const res = await api.retrainMlModel(req);
      setTrainingStatus(res.message);
      const poll = setInterval(async () => {
        try {
          const status = await api.getMlRetrainStatus(req.name || `model_${Date.now()}`);
          setTrainingStatus(status.message);
          if (status.status === "completed" || status.status === "unknown") {
            clearInterval(poll);
            fetchModels();
          }
        } catch {
          clearInterval(poll);
        }
      }, 3000);
    } catch (e: unknown) {
      setTrainingStatus(`Error: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const handleDelete = async (name: string) => {
    try {
      await api.deleteMlModel(name);
      fetchModels();
    } catch (e: unknown) {
      alert(`Delete failed: ${e instanceof Error ? e.message : String(e)}`);
    }
  };

  const toggleBool = (key: string) => {
    setReq((prev) => ({ ...prev, [key]: !(prev as unknown as Record<string, boolean>)[key] }));
  };

  const sel: React.CSSProperties = {
    padding: "4px 8px",
    background: colors.inputBg,
    color: colors.text,
    border: `1px solid ${colors.border}`,
    borderRadius: 4,
    fontSize: 13,
  };

  const inp: React.CSSProperties = {
    ...sel,
    width: 160,
  };

  const chk: React.CSSProperties = {
    cursor: "pointer",
  };

  const labelStyle: React.CSSProperties = {
    fontSize: 12,
    color: colors.text,
    opacity: 0.8,
    display: "flex",
    alignItems: "center",
    gap: 4,
  };

  return (
    <div>
      <h3 style={{ margin: "0 0 16px" }}>ML Lab</h3>

      {/* Retrain form */}
      <div
        style={{
          background: colors.surface,
          border: `1px solid ${colors.border}`,
          borderRadius: 8,
          padding: 16,
          marginBottom: 16,
        }}
      >
        <h4 style={{ margin: "0 0 12px" }}>Retrain Model</h4>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "flex-end" }}>
          <div>
            <div style={labelStyle}>Symbols</div>
            <input style={inp} value={req.symbols} onChange={(e) => setReq({ ...req, symbols: e.target.value })} />
          </div>
          <div>
            <div style={labelStyle}>Years</div>
            <input
              style={{ ...inp, width: 60 }}
              type="number"
              value={req.years}
              onChange={(e) => setReq({ ...req, years: +e.target.value })}
            />
          </div>
          <div>
            <div style={labelStyle}>Name</div>
            <input
              style={inp}
              placeholder="auto-name"
              value={req.name}
              onChange={(e) => setReq({ ...req, name: e.target.value })}
            />
          </div>
          <div>
            <div style={labelStyle}>Model Types</div>
            <select style={sel} value={req.model_types} onChange={(e) => setReq({ ...req, model_types: e.target.value })}>
              <option value="rf,gbt">rf,gbt</option>
              <option value="rf,gbt,xgb,lgb">rf,gbt,xgb,lgb</option>
              <option value="xgb,lgb">xgb,lgb</option>
              <option value="rf,gbt,sgd,mlp">rf,gbt,sgd,mlp</option>
              <option value="all">all</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>Labeling</div>
            <select style={sel} value={req.labeling} onChange={(e) => setReq({ ...req, labeling: e.target.value })}>
              <option value="next_bar">next_bar</option>
              <option value="triple_barrier">triple_barrier</option>
            </select>
          </div>
          <div>
            <div style={labelStyle}>Horizon</div>
            <input
              style={{ ...inp, width: 60 }}
              type="number"
              min={1}
              value={req.forecast_horizon}
              onChange={(e) => setReq({ ...req, forecast_horizon: +e.target.value })}
            />
          </div>
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 10 }}>
          {([
            ["grid_search", "Grid Search"],
            ["walk_forward", "Walk-Forward"],
            ["stacking", "Stacking"],
            ["meta_labeling", "Meta-Label"],
            ["regularize", "Regularize"],
            ["kelly", "Kelly"],
            ["auto_threshold", "Auto-Thresh"],
            ["beat_baselines", "Beat Base"],
            ["regime_aware", "Regime"],
          ] as const).map(([key, label]) => (
            <label key={key} style={labelStyle}>
              <input
                type="checkbox"
                style={chk}
                checked={!!(req as unknown as Record<string, unknown>)[key as string]}
                onChange={() => toggleBool(key as keyof MlRetrainRequest)}
              />
              {label}
            </label>
          ))}
        </div>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 12, marginTop: 8 }}>
          <div>
            <div style={labelStyle}>Context Symbols</div>
            <input
              style={inp}
              placeholder="SPY,VOO"
              value={req.context_symbols ?? ""}
              onChange={(e) => setReq({ ...req, context_symbols: e.target.value || undefined })}
            />
          </div>
          <div>
            <div style={labelStyle}>Multi-Horizon</div>
            <input
              style={{ ...inp, width: 100 }}
              placeholder="1,5,21"
              value={req.multi_horizon ?? ""}
              onChange={(e) => setReq({ ...req, multi_horizon: e.target.value || undefined })}
            />
          </div>
          <div>
            <div style={labelStyle}>Prune %</div>
            <input
              style={{ ...inp, width: 60 }}
              type="number"
              min={0}
              max={0.5}
              step={0.05}
              value={req.prune}
              onChange={(e) => setReq({ ...req, prune: +e.target.value })}
            />
          </div>
        </div>
        <div style={{ marginTop: 12, display: "flex", gap: 8, alignItems: "center" }}>
          <button
            onClick={handleRetrain}
            style={{
              padding: "8px 24px",
              background: colors.primary,
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontWeight: 600,
            }}
          >
            Start Retrain
          </button>
          {trainingStatus && <span style={{ fontSize: 12, color: colors.text }}>{trainingStatus}</span>}
        </div>
      </div>

      {/* Model list */}
      <h4 style={{ margin: "0 0 8px" }}>Trained Models</h4>
      {loading ? (
        <p style={{ color: colors.text, opacity: 0.6 }}>Loading...</p>
      ) : models.length === 0 ? (
        <p style={{ color: colors.text, opacity: 0.6 }}>No models trained yet.</p>
      ) : (
        models.map((m) => <ModelCard key={`${m.name}-${m.version}`} model={m} onDelete={() => handleDelete(m.name)} />)
      )}
    </div>
  );
}
