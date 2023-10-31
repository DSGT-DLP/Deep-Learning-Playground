import HtmlTooltip from "@/common/components/HtmlTooltip";
import InfoIcon from "@mui/icons-material/Info";
import {
  Button,
  Card,
  Divider,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  Edge,
  Handle,
  Node,
  NodeTypes,
  Position,
  addEdge,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";
import {
  ALL_LAYERS,
  DEFAULT_LAYERS,
  STEP_SETTINGS,
} from "../constants/tabularConstants";
import { ParameterData } from "../types/tabularTypes";
import assert from "assert";
import { nanoid } from "nanoid/non-secure";
import { toast } from "react-toastify";

interface TabularFlowProps {
  setLayers?: (layers: ParameterData["layers"]) => void;
}

export default function TabularFlow(props: TabularFlowProps) {
  const { setLayers } = props;

  const initialNodes: Node<LayerNodeData>[] = [
    ROOT_NODE,
    ...DEFAULT_LAYERS.IRIS.map((layer, i) => ({
      id: `${layer.value}-${i}`,
      type: "textUpdater",
      position: {
        x: DEFAULT_X_POSITION,
        y: (i + 1) * 100,
      },
      data: {
        label: STEP_SETTINGS.PARAMETERS.layers[layer.value].label,
        value: layer.value,
        parameters: layer.parameters.slice(),
        onChange: onChange,
      },
    })),
  ];
  const initialEdges: Edge[] = createInitialEdges();

  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  useEffect(() => {
    if (setLayers) {
      const layers = handleExportLayers();
      setLayers(layers);
    }
  }, [nodes, edges]);

  function handleExportLayers(): ParameterData["layers"] {
    const layers: ParameterData["layers"] = [];
    const visited = new Set<string>(["root"]);

    const directedEdges: Record<string, Node<LayerNodeData>> = {};
    edges.forEach((edge) => {
      const sourceNode = nodes.find((node) => node.id === edge.source);
      const targetNode = nodes.find((node) => node.id === edge.target);
      if (sourceNode && targetNode) {
        directedEdges[sourceNode.id] = targetNode;
      }
    });

    let nextNode = directedEdges["root"];

    // appending all layers from the graph
    while (nextNode) {
      assert(nextNode.data.value !== "root");

      if (visited.has(nextNode.id)) {
        toast.error("Cycle detected in layers");
        return layers;
      }
      visited.add(nextNode.id);
      layers.push({
        value: nextNode.data.value,
        parameters: nextNode.data.parameters || [],
      });
      nextNode = directedEdges[nextNode.id];
    }

    return layers;
  }

  function onChange(args: OnChangeArgs) {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id !== args.id) return node;
        if (!node.data.parameters) return node;

        node.data.parameters[args.parameterIndex] = args.newValue;
        return node;
      })
    );
  }

  return (
    <>
      <Paper sx={{ mb: 2 }}>
        <Stack alignItems="center" spacing={2} padding={2}>
          <Typography variant="h2" fontSize={25}>
            Layers
          </Typography>
          <Stack
            direction="row"
            gap={1}
            flexWrap="wrap"
            justifyContent="space-between"
          >
            {STEP_SETTINGS.PARAMETERS.layerValues.map((value) => (
              <Button
                key={value}
                variant="outlined"
                color="primary"
                onClick={() => {
                  setNodes((cur) => [
                    ...cur,
                    {
                      id: `${value}-${nanoid()}`,
                      type: "textUpdater",
                      position: {
                        x: DEFAULT_X_POSITION,
                        y: Math.random() * 50,
                      },
                      data: {
                        label: STEP_SETTINGS.PARAMETERS.layers[value].label,
                        value: value,
                        onChange: onChange,
                        parameters: STEP_SETTINGS.PARAMETERS.layers[
                          value
                        ].parameters.map(() => 0),
                      },
                    },
                  ]);
                }}
              >
                {value}
              </Button>
            ))}
          </Stack>
        </Stack>
      </Paper>
      <div style={{ width: "100%", height: 500 }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={(params) => setEdges((eds) => addEdge(params, eds))}
          nodeTypes={nodeTypes}
        >
          <Controls />
          <Background gap={12} size={1} variant={BackgroundVariant.Dots} />
        </ReactFlow>
      </div>
    </>
  );
}

interface TextUpdaterNodeProps {
  id: string;
  data: LayerNodeData;
  isConnectable: boolean;
}

function TextUpdaterNode(props: TextUpdaterNodeProps) {
  const { data, isConnectable, id } = props;
  assert(data.value !== "root");
  const layer = STEP_SETTINGS.PARAMETERS.layers[data.value];

  return (
    <>
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
      />
      <div>
        <Card sx={{ p: 3 }}>
          <Stack
            direction="row"
            justifyContent="space-between"
            alignItems="center"
            spacing={3}
          >
            <HtmlTooltip
              title={
                <React.Fragment>
                  <Typography color="inherit">{layer.label}</Typography>
                  {layer.description}
                </React.Fragment>
              }
            >
              <InfoIcon>Info</InfoIcon>
            </HtmlTooltip>

            <Typography variant="h3" fontSize={18}>
              {layer.label}
            </Typography>
            <Stack direction="row" alignItems="center" spacing={3}>
              <Stack
                direction="row"
                alignItems="center"
                justifyContent="flex-end"
                spacing={2}
                divider={<Divider orientation="vertical" flexItem />}
              >
                {layer.parameters.map((parameter, index) => (
                  <TextField
                    onBlur={(e) =>
                      data.onChange({
                        id: id,
                        newValue: Number(e.target.value),
                        parameterIndex: index,
                      })
                    }
                    key={index}
                    label={parameter.label}
                    defaultValue={data.parameters?.[index]}
                    size="small"
                    type={parameter.type}
                    required
                  />
                ))}
              </Stack>
            </Stack>
          </Stack>
        </Card>
      </div>
      <Handle
        type="source"
        position={Position.Bottom}
        id="b"
        isConnectable={isConnectable}
      />
    </>
  );
}

interface OnChangeArgs {
  id: string;
  newValue: number;
  parameterIndex: number;
}

interface LayerNodeData {
  label:
    | (typeof STEP_SETTINGS.PARAMETERS.layers)[ALL_LAYERS]["label"]
    | "Beginning";
  value: ALL_LAYERS | "root";
  parameters?: number[];
  onChange: (args: OnChangeArgs) => void;
}

const DEFAULT_X_POSITION = 10;

const ROOT_NODE: Node<LayerNodeData> = {
  id: `root`,
  type: "input",
  position: { x: DEFAULT_X_POSITION, y: 0 },
  deletable: false,
  data: {
    label: "Beginning",
    value: "root",
    parameters: [],
    onChange: () => undefined,
  },
};
const nodeTypes: NodeTypes = { textUpdater: TextUpdaterNode };

function createInitialEdges(): Edge[] {
  const edges: Edge[] = [];
  const defaultLayers = DEFAULT_LAYERS.IRIS;

  // connecting root to first layer
  edges.push({
    id: `eroot-0`,
    source: "root",
    target: `${defaultLayers[0].value}-0`,
  });

  // connecting all layers
  for (let i = 0; i < defaultLayers.length - 1; i++) {
    edges.push({
      id: `e${i}-${i + 1}`,
      source: `${defaultLayers[i].value}-${i}`,
      target: `${defaultLayers[i + 1].value}-${i + 1}`,
    });
  }
  return edges;
}
