import { useContext, useEffect, useState, useCallback, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import DJClientContext from '../../providers/djclient';
import MetricFlowGraph from './MetricFlowGraph';
import SelectionPanel from './SelectionPanel';
import {
  PreAggDetailsPanel,
  MetricDetailsPanel,
  QueryOverviewPanel,
} from './PreAggDetailsPanel';
import './styles.css';

/**
 * Helper to normalize grain columns to short names for comparison
 * "default.date_dim.date_id" -> "date_id"
 */
function normalizeGrain(grainCols) {
  return (grainCols || [])
    .map(col => col.split('.').pop())
    .sort()
    .join(',');
}

export function QueryPlannerPage() {
  const djClient = useContext(DJClientContext).DataJunctionAPI;
  const location = useLocation();
  const navigate = useNavigate();

  // Available options
  const [metrics, setMetrics] = useState([]);
  const [commonDimensions, setCommonDimensions] = useState([]);

  // Selection state - initialized from URL params
  const [selectedMetrics, setSelectedMetrics] = useState([]);
  const [selectedDimensions, setSelectedDimensions] = useState([]);

  // Track if we've initialized from URL (to avoid overwriting URL on first render)
  const initializedFromUrl = useRef(false);
  const pendingDimensionsFromUrl = useRef([]);

  // Results state
  const [measuresResult, setMeasuresResult] = useState(null);
  const [metricsResult, setMetricsResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dimensionsLoading, setDimensionsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Node selection for details panel
  const [selectedNode, setSelectedNode] = useState(null);

  // Materialization state - map of grain_key -> pre-agg info
  const [plannedPreaggs, setPlannedPreaggs] = useState({});

  // Initialize selection from URL params on mount
  useEffect(() => {
    const params = new URLSearchParams(location.search);
    const urlMetrics = params.get('metrics')?.split(',').filter(Boolean) || [];
    const urlDimensions =
      params.get('dimensions')?.split(',').filter(Boolean) || [];

    if (urlMetrics.length > 0) {
      setSelectedMetrics(urlMetrics);
      // Store dimensions to apply after commonDimensions are loaded
      if (urlDimensions.length > 0) {
        pendingDimensionsFromUrl.current = urlDimensions;
      }
      initializedFromUrl.current = true;
    }
  }, []); // Only run on mount

  // Update URL when selection changes
  useEffect(() => {
    // Skip the first render if we just initialized from URL
    if (!initializedFromUrl.current && selectedMetrics.length === 0) {
      return;
    }

    const params = new URLSearchParams();
    if (selectedMetrics.length > 0) {
      params.set('metrics', selectedMetrics.join(','));
    }
    if (selectedDimensions.length > 0) {
      params.set('dimensions', selectedDimensions.join(','));
    }

    const newSearch = params.toString();
    const currentSearch = location.search.replace(/^\?/, '');

    // Only update if different (avoid unnecessary history entries)
    if (newSearch !== currentSearch) {
      navigate(
        {
          pathname: location.pathname,
          search: newSearch ? `?${newSearch}` : '',
        },
        { replace: true },
      );
    }
  }, [selectedMetrics, selectedDimensions, location.pathname, navigate]);

  // Get metrics list on mount
  useEffect(() => {
    const fetchData = async () => {
      const metricsList = await djClient.metrics();
      setMetrics(metricsList);
    };
    fetchData().catch(console.error);
  }, [djClient]);

  // Get common dimensions when metrics change
  useEffect(() => {
    const fetchData = async () => {
      if (selectedMetrics.length > 0) {
        setDimensionsLoading(true);
        try {
          const dims = await djClient.commonDimensions(selectedMetrics);
          setCommonDimensions(dims);

          // Apply pending dimensions from URL if we have them
          if (pendingDimensionsFromUrl.current.length > 0) {
            const validDimNames = dims.map(d => d.name);
            const validPending = pendingDimensionsFromUrl.current.filter(d =>
              validDimNames.includes(d),
            );
            if (validPending.length > 0) {
              setSelectedDimensions(validPending);
            }
            pendingDimensionsFromUrl.current = []; // Clear after applying
          }
        } catch (err) {
          console.error('Failed to fetch dimensions:', err);
          setCommonDimensions([]);
        }
        setDimensionsLoading(false);
      } else {
        setCommonDimensions([]);
        setSelectedDimensions([]);
      }
    };
    fetchData().catch(console.error);
  }, [selectedMetrics, djClient]);

  // Clear dimension selections that are no longer valid
  useEffect(() => {
    const validDimNames = commonDimensions.map(d => d.name);
    const validSelections = selectedDimensions.filter(d =>
      validDimNames.includes(d),
    );
    if (validSelections.length !== selectedDimensions.length) {
      setSelectedDimensions(validSelections);
    }
  }, [commonDimensions, selectedDimensions]);

  // Fetch V3 measures and metrics SQL when selection changes
  useEffect(() => {
    const fetchData = async () => {
      if (selectedMetrics.length > 0 && selectedDimensions.length > 0) {
        setLoading(true);
        setError(null);
        setSelectedNode(null);
        try {
          // Fetch both measures and metrics SQL in parallel
          const [measures, metrics] = await Promise.all([
            djClient.measuresV3(selectedMetrics, selectedDimensions),
            djClient.metricsV3(selectedMetrics, selectedDimensions),
          ]);
          setMeasuresResult(measures);
          setMetricsResult(metrics);
        } catch (err) {
          setError(err.message || 'Failed to fetch data');
          setMeasuresResult(null);
          setMetricsResult(null);
        }
        setLoading(false);
      } else {
        setMeasuresResult(null);
        setMetricsResult(null);
      }
    };
    fetchData().catch(console.error);
  }, [djClient, selectedMetrics, selectedDimensions]);

  // Fetch existing pre-aggregations for the grain groups
  useEffect(() => {
    const fetchExistingPreaggs = async () => {
      if (!measuresResult?.grain_groups?.length) {
        setPlannedPreaggs({});
        return;
      }

      // Get unique node names from grain groups
      const parentNames = measuresResult.grain_groups.map(gg => gg.parent_name);
      // Use Array.from instead of spread - spread on Set seems broken in this env
      const nodeNames = Array.from(new Set(parentNames.filter(Boolean)));

      if (!nodeNames.length) {
        return;
      }

      try {
        // Fetch pre-aggs for each node in parallel
        const preaggResults = await Promise.all(
          nodeNames.map(nodeName =>
            djClient.listPreaggs({ node_name: nodeName }),
          ),
        );

        // Build lookup map by normalized grain key
        // Key format: "parent_name|short_grain_col1,short_grain_col2"
        const newPreaggs = {};
        preaggResults.forEach(result => {
          // API returns paginated response with `items` array
          const preaggs = result.items || result.pre_aggregations || [];
          if (Array.isArray(preaggs)) {
            preaggs.forEach(preagg => {
              // Normalize grain_columns to short names for matching
              const grainKey = `${preagg.node_name}|${normalizeGrain(
                preagg.grain_columns,
              )}`;
              // Only keep the first (or latest) pre-agg for each grain key
              if (!newPreaggs[grainKey]) {
                newPreaggs[grainKey] = preagg;
              }
            });
          }
        });

        setPlannedPreaggs(newPreaggs);
      } catch (err) {
        console.error('Failed to fetch existing pre-aggs:', err);
      }
    };

    fetchExistingPreaggs();
  }, [measuresResult, djClient]);

  const handleMetricsChange = useCallback(newMetrics => {
    setSelectedMetrics(newMetrics);
    setSelectedNode(null);
  }, []);

  const handleDimensionsChange = useCallback(newDimensions => {
    setSelectedDimensions(newDimensions);
    setSelectedNode(null);
  }, []);

  const handleNodeSelect = useCallback(node => {
    setSelectedNode(node);
  }, []);

  const handleClosePanel = useCallback(() => {
    setSelectedNode(null);
  }, []);

  // Handle planning/saving a new materialization configuration
  const handlePlanMaterialization = useCallback(
    async (grainGroup, config) => {
      try {
        // Call the plan endpoint with current metrics/dims and config
        const result = await djClient.planPreaggs(
          selectedMetrics,
          selectedDimensions,
          config.strategy,
          config.schedule,
          config.lookbackWindow,
        );

        // API returns paginated response with `items` or `pre_aggregations`
        const preaggs = result.items || result.pre_aggregations || [];
        if (preaggs.length > 0) {
          // Update our local state with the planned pre-aggs
          const newPreaggs = { ...plannedPreaggs };
          preaggs.forEach(preagg => {
            // Use normalized grain for consistent key matching
            const grainKey = `${preagg.node_name}|${normalizeGrain(
              preagg.grain_columns,
            )}`;
            newPreaggs[grainKey] = preagg;
          });
          setPlannedPreaggs(newPreaggs);
        }

        return result;
      } catch (err) {
        console.error('Failed to plan materialization:', err);
        throw err;
      }
    },
    [djClient, selectedMetrics, selectedDimensions, plannedPreaggs],
  );

  // Handle triggering materialization for a specific pre-agg
  const handleTriggerMaterialization = useCallback(
    async preaggId => {
      try {
        const result = await djClient.materializePreagg(preaggId);

        if (result.id) {
          // Update the pre-agg status in our local state
          setPlannedPreaggs(prev => {
            const updated = { ...prev };
            // Find and update the matching pre-agg
            for (const key in updated) {
              if (updated[key].id === preaggId) {
                updated[key] = { ...updated[key], status: 'running' };
                break;
              }
            }
            return updated;
          });
        }
      } catch (err) {
        console.error('Failed to trigger materialization:', err);
        // Could add error state/toast here
      }
    },
    [djClient],
  );

  return (
    <div className="planner-page">
      {/* Header */}
      <header className="planner-header">
        <div className="planner-header-content">
          <h1>Query Planner</h1>
          <p>Explore metrics and dimensions and plan materializations</p>
        </div>
        {error && <div className="header-error">{error}</div>}
      </header>

      {/* Three-column layout */}
      <div className="planner-layout">
        {/* Left: Selection Panel */}
        <aside className="planner-selection">
          <SelectionPanel
            metrics={metrics}
            selectedMetrics={selectedMetrics}
            onMetricsChange={handleMetricsChange}
            dimensions={commonDimensions}
            selectedDimensions={selectedDimensions}
            onDimensionsChange={handleDimensionsChange}
            loading={dimensionsLoading}
          />
        </aside>

        {/* Center: Graph */}
        <main className="planner-graph">
          {loading ? (
            <div className="graph-loading">
              <div className="loading-spinner" />
              <span>Building data flow...</span>
            </div>
          ) : measuresResult ? (
            <>
              <div className="graph-header">
                <span className="graph-stats">
                  {measuresResult.grain_groups?.length || 0} pre-aggregations →{' '}
                  {measuresResult.metric_formulas?.length || 0} metrics
                </span>
              </div>
              <MetricFlowGraph
                grainGroups={measuresResult.grain_groups}
                metricFormulas={measuresResult.metric_formulas}
                selectedNode={selectedNode}
                onNodeSelect={handleNodeSelect}
              />
            </>
          ) : (
            <div className="graph-empty">
              <div className="empty-icon">⊞</div>
              <h3>Select Metrics & Dimensions</h3>
              <p>
                Choose metrics from the left panel, then select dimensions to
                see how they decompose into pre-aggregations.
              </p>
            </div>
          )}
        </main>

        {/* Right: Details Panel */}
        <aside className="planner-details">
          {selectedNode?.type === 'preagg' ? (
            <PreAggDetailsPanel
              preAgg={selectedNode.data}
              metricFormulas={measuresResult?.metric_formulas}
              onClose={handleClosePanel}
            />
          ) : selectedNode?.type === 'metric' ? (
            <MetricDetailsPanel
              metric={selectedNode.data}
              grainGroups={measuresResult?.grain_groups}
              onClose={handleClosePanel}
            />
          ) : (
            <QueryOverviewPanel
              measuresResult={measuresResult}
              metricsResult={metricsResult}
              selectedMetrics={selectedMetrics}
              selectedDimensions={selectedDimensions}
              plannedPreaggs={plannedPreaggs}
              onPlanMaterialization={handlePlanMaterialization}
              onTriggerMaterialization={handleTriggerMaterialization}
            />
          )}
        </aside>
      </div>
    </div>
  );
}

export default QueryPlannerPage;
