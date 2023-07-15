import {
  Stack,
  Paper,
  Typography,
  Button,
  Tooltip,
  tooltipClasses,
  IconButton,
  Card,
  TextField,
  Divider,
} from "@mui/material";
import React, { useCallback, useMemo } from "react";
import ReactFlow, {
  MiniMap,
  Controls,
  Background,
  useNodesState,
  useEdgesState,
  addEdge,
  BackgroundVariant,
  Handle,
  Position,
  NodeTypes,
  Node,
  Edge,
} from "reactflow";
import "reactflow/dist/style.css";
import { STEP_SETTINGS } from "../constants/tabularConstants";
import { styled } from "@mui/material/styles";
import { Controller } from "react-hook-form";
import DeleteIcon from "@mui/icons-material/Delete";
import InfoIcon from "@mui/icons-material/Info";

type ALL_LAYERS = keyof typeof STEP_SETTINGS.PARAMETERS.layers;

interface NodeData {
  label: (typeof STEP_SETTINGS.PARAMETERS.layers)[ALL_LAYERS]["label"];
  value: ALL_LAYERS;
}

const initialNodes: Node<NodeData>[] = [
  {
    id: `LINEAR-${Math.random() * 100}`,
    type: "textUpdater",
    position: { x: 0, y: 0 },
    data: { label: "Linear", value: "LINEAR" },
  },
  {
    id: `RELU-${Math.random() * 100}`,
    type: "input",
    position: { x: 0, y: 100 },
    data: { label: "ReLU", value: "RELU" },
  },
];
const initialEdges: Edge[] = [{ id: "e1-2", source: "1", target: "2" }];

export default function TabularDnd() {
  const [nodes, setNodes, onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  console.log(nodes);

  const nodeTypes = useMemo(() => ({ textUpdater: TextUpdaterNode }), []);

  return (
    <>
      <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
        <Stack alignItems="center" spacing={2} padding={2}>
          <Typography variant="h2" fontSize={25}>
            Layers
          </Typography>
          <Stack direction="row" spacing={3}>
            {STEP_SETTINGS.PARAMETERS.layerValues.map((value) => (
              <Button
                key={value}
                variant="contained"
                color="primary"
                onClick={() =>
                  setNodes((cur) => [
                    ...cur,
                    {
                      id: `${value}-${Math.random() * 100}`,
                      position: {
                        x: Math.random() * 50,
                        y: Math.random() * 50,
                      },
                      data: { label: value, value: value },
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
      <div style={{ width: "100%", height: "50vh" }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onConnect={(params) => setEdges((eds) => addEdge(params, eds))}
          nodeTypes={nodeTypes}
        >
          <Controls />
          <MiniMap />
          <Background gap={12} size={1} variant={BackgroundVariant.Dots} />
        </ReactFlow>
      </div>
    </>
  );
}

function TextUpdaterNode({
  data,
  isConnectable,
}: {
  data: NodeData;
  isConnectable: boolean;
}) {
  const onChange = useCallback((evt) => {
    console.log(evt.target.value);
  }, []);

  return (
    <div
      style={{
        height: "50px",
        border: "1px solid #eee",
        padding: "5px",
        borderRadius: "5px",
        background: "white",
      }}
    >
      <Handle
        type="target"
        position={Position.Top}
        isConnectable={isConnectable}
      />
      <div>
        <Card sx={{ p: 3 }} style={{ display: "inline-block" }}>
          <Stack
            direction={"row"}
            justifyContent={"space-between"}
            alignItems={"center"}
            spacing={3}
          >
            <HtmlTooltip
              title={
                <React.Fragment>
                  <Typography color="inherit">
                    {STEP_SETTINGS.PARAMETERS.layers[data.value].label}
                  </Typography>
                  {STEP_SETTINGS.PARAMETERS.layers[data.value].description}
                </React.Fragment>
              }
            >
              <InfoIcon>Info</InfoIcon>
            </HtmlTooltip>

            <Typography variant="h3" fontSize={18}>
              {STEP_SETTINGS.PARAMETERS.layers[data.value].label}
            </Typography>
            <Stack direction={"row"} alignItems={"center"} spacing={3}>
              <Stack
                direction={"row"}
                alignItems={"center"}
                justifyContent={"flex-end"}
                spacing={2}
                divider={<Divider orientation="vertical" flexItem />}
              >
                {STEP_SETTINGS.PARAMETERS.layers[data.value].parameters.map(
                  (parameter, index) => (
                    <div key={index} data-no-dnd>
                      {formProps ? (
                        <Controller
                          name={`layers.${formProps.index}.parameters.${index}`}
                          control={formProps.control}
                          rules={{ required: true }}
                          render={({ field: { onChange, value } }) => (
                            <TextField
                              label={parameter.label}
                              size={"small"}
                              type={parameter.type}
                              onChange={onChange}
                              value={value}
                              required
                              error={
                                formProps.errors.layers?.[formProps.index]
                                  ?.parameters?.[index]
                                  ? true
                                  : false
                              }
                            />
                          )}
                        />
                      ) : (
                        <TextField
                          label={parameter.label}
                          size={"small"}
                          type={parameter.type}
                          value={""}
                          required
                        />
                      )}
                    </div>
                  )
                )}
              </Stack>
              <div data-no-dnd>
                <IconButton onClick={formProps?.remove}>
                  <DeleteIcon />
                </IconButton>
              </div>
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
    </div>
  );
}

const HtmlTooltip = styled(
  ({
    className,
    title,
    children,
    ...props
  }: {
    className?: string;
    children: React.ReactElement;
    title: React.ReactNode;
  }) => (
    <Tooltip title={title} {...props} classes={{ popper: className }}>
      {children}
    </Tooltip>
  )
)(({ theme }) => ({
  [`& .${tooltipClasses.tooltip}`]: {
    backgroundColor: "rgba(255, 255, 255, 0.95)",
    color: "rgba(0, 0, 0, 0.87)",
    maxWidth: 220,
    fontSize: theme.typography.pxToRem(12),
    border: "none",
  },
}));
