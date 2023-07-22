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
import React, { useCallback } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  Controls,
  Edge,
  Handle,
  MiniMap,
  Node,
  NodeTypes,
  Position,
  addEdge,
  useEdgesState,
  useNodesState,
} from "reactflow";
import "reactflow/dist/style.css";
import { STEP_SETTINGS } from "../constants/tabularConstants";
import { ParameterData } from "../types/tabularTypes";

type ALL_LAYERS = keyof typeof STEP_SETTINGS.PARAMETERS.layers;

interface OnChangeArgs {
  id: string;
  newValue: number;
  parameterIndex: number;
}

interface NodeData {
  label:
    | (typeof STEP_SETTINGS.PARAMETERS.layers)[ALL_LAYERS]["label"]
    | "Start";
  value: ALL_LAYERS | "root";
  parameters?: number[];
  onChange: (args: OnChangeArgs) => void;
}

const initialNodes: Node<NodeData>[] = [
  {
    id: `root`,
    type: "input",
    position: { x: 0, y: 0 },
    deletable: false,
    data: {
      label: "Start",
      value: "root",
      parameters: [],
      onChange: () => undefined,
    },
  },
];
const initialEdges: Edge[] = [{ id: "e1-2", source: "1", target: "2" }];
const nodeTypes: NodeTypes = { textUpdater: TextUpdaterNode };

export default function TabularDnd() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  const onChange = useCallback((args: OnChangeArgs) => {
    setNodes((nds) =>
      nds.map((node) => {
        if (node.id !== args.id) return node;
        if (!node.data.parameters) return node;

        node.data.parameters[args.parameterIndex] = args.newValue;
        return node;
      })
    );
  }, []);

  return (
    <>
      <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
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
                onClick={() =>
                  setNodes((cur) => [
                    ...cur,
                    {
                      id: `${value}-${Math.random() * 100}`,
                      type: "textUpdater",
                      position: {
                        x: Math.random() * 50,
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
                  ])
                }
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
  data: NodeData;
  isConnectable: boolean;
}

function TextUpdaterNode(props: TextUpdaterNodeProps) {
  const { data, isConnectable, id } = props;
  const layer = STEP_SETTINGS.PARAMETERS.layers[data.value as ALL_LAYERS];

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

export function exportLayers(): ParameterData["layers"][] | undefined {
  
  
  return [];
}
