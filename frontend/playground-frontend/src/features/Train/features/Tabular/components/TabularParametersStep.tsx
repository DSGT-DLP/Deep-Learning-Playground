import React, { useMemo, useState } from "react";
import { useGetColumnsFromDatasetQuery } from "@/features/Train/redux/trainspaceApi";
import { useAppSelector } from "@/common/redux/hooks";
import {
  Autocomplete,
  Card,
  Container,
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormLabel,
  IconButton,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Slider,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import { Control, Controller, useFieldArray, useForm } from "react-hook-form";
import { ParameterData, TrainspaceData } from "../types/tabularTypes";
import { STEP_SETTINGS } from "../constants/tabularConstants";
import { CSS } from "@dnd-kit/utilities";
import {
  SortableContext,
  arrayMove,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import {
  Active,
  DndContext,
  DragEndEvent,
  DragOverEvent,
  DragOverlay,
  closestCenter,
  useDraggable,
  useSensors,
} from "@dnd-kit/core";
import DeleteIcon from "@mui/icons-material/Delete";
import {
  useCustomKeyboardSensor,
  useCustomPointerSensor,
} from "@/common/utils/dndHelpers";
import ClientOnlyPortal from "@/common/components/ClientOnlyPortal";

const TabularParametersStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData) => void
  ) => React.ReactNode;
}) => {
  const trainspace = useAppSelector(
    (state) =>
      state.trainspace.current as TrainspaceData<"PARAMETERS"> | undefined
  );
  if (!trainspace) return <></>;
  const { data, refetch } = useGetColumnsFromDatasetQuery({
    dataSource: "TABULAR",
    dataset: trainspace.datasetData,
  });
  if (!data) return <></>;

  return <TabularStepsInner trainspace={trainspace} data={data} />;
};

const TabularStepsInner = ({
  trainspace,
  data,
}: {
  trainspace: TrainspaceData<"PARAMETERS">;
  data: string[];
}) => {
  const {
    handleSubmit,
    formState: { errors },
    control,
    register,
    reset,
  } = useForm<ParameterData>({
    defaultValues: {
      problemType: "CLASSIFICATION",
      criterion: "CELOSS",
      optimizerName: "SGD",
      shuffle: true,
      epochs: 5,
      batchSize: 20,
      testSize: 0.2,
      layers: [
        {
          value: "LINEAR",
          parameters: [10, 3],
        },
        {
          value: "RELU",
          parameters: [],
        },
        {
          value: "LINEAR",
          parameters: [3, 10],
        },
        {
          value: "SOFTMAX",
          parameters: [-1],
        },
      ],
    },
  });
  return (
    <Stack spacing={3}>
      <Autocomplete
        disableClearable
        autoComplete
        autoHighlight
        renderInput={(params) => (
          <TextField
            {...params}
            required
            label="Target Column"
            placeholder="Target"
          />
        )}
        options={data}
      />
      <Autocomplete
        multiple
        autoComplete
        autoHighlight
        disableCloseOnSelect
        renderInput={(params) => (
          <TextField
            {...params}
            required
            label="Feature Columns"
            placeholder="Features"
          />
        )}
        options={data}
      />
      <FormControl>
        <FormLabel>Problem Type</FormLabel>
        <Controller
          name="problemType"
          control={control}
          rules={{ required: true }}
          render={({ field: { onChange, value } }) => (
            <RadioGroup row value={value} onChange={onChange}>
              {STEP_SETTINGS["PARAMETERS"].problemTypes.map((problemType) => (
                <FormControlLabel
                  key={problemType.value}
                  value={problemType.value}
                  control={<Radio />}
                  label={problemType.label}
                />
              ))}
            </RadioGroup>
          )}
        />
      </FormControl>
      <Controller
        name="criterion"
        control={control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <TextField select label="Criterion" onChange={onChange} value={value}>
            {STEP_SETTINGS["PARAMETERS"].criterions.map((criterion) => (
              <MenuItem key={criterion.value} value={criterion.value}>
                {criterion.label}
              </MenuItem>
            ))}
          </TextField>
        )}
      />
      <Controller
        name="optimizerName"
        control={control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <TextField select label="Optimizer" onChange={onChange} value={value}>
            {STEP_SETTINGS["PARAMETERS"].optimizers.map((optimizer) => (
              <MenuItem key={optimizer.value} value={optimizer.value}>
                {optimizer.label}
              </MenuItem>
            ))}
          </TextField>
        )}
      />
      <FormGroup>
        <Controller
          name="shuffle"
          control={control}
          rules={{ required: true }}
          render={({ field: { onChange, value } }) => (
            <FormControlLabel
              control={
                <Switch value={value} onChange={onChange} defaultChecked />
              }
              label="Shuffle"
            />
          )}
        />
      </FormGroup>
      <Controller
        name="epochs"
        control={control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <TextField
            label="Epochs"
            type={"number"}
            onChange={onChange}
            value={value}
          />
        )}
      />
      <Controller
        name="batchSize"
        control={control}
        rules={{ required: true }}
        render={({ field: { onChange, value } }) => (
          <TextField
            label="Batch Size"
            type={"number"}
            onChange={onChange}
            value={value}
          />
        )}
      />
      <FormControl>
        <FormLabel>Test Size</FormLabel>
        <Controller
          name="testSize"
          control={control}
          rules={{ required: true }}
          render={({ field: { onChange, value } }) => (
            <Slider
              step={0.01}
              value={value}
              onChange={onChange}
              min={0}
              max={1}
              valueLabelDisplay="auto"
            />
          )}
        />
      </FormControl>
      <LayersDnd control={control} />
    </Stack>
  );
};

const LayersDnd = ({
  control,
}: {
  control: Control<ParameterData, unknown>;
}) => {
  const {
    fields,
    move: fieldsMove,
    insert: fieldsInsert,
    remove: fieldsRemove,
  } = useFieldArray({
    control: control,
    name: "layers",
  });
  const generateDragId = () => Math.floor(Math.random() * Date.now());
  const [layerIdsList, setLayerIdsList] = useState<number[]>(
    fields.map(() => generateDragId())
  );
  const [layerInventoryIdsMap, setLayerInventoryIdsMap] = useState<{
    [layerValue: string]: number;
  }>(
    Object.fromEntries(
      STEP_SETTINGS.PARAMETERS.layerValues.map((layerValue) => [
        layerValue,
        generateDragId(),
      ])
    )
  );
  const [active, setActive] = useState<Active | null>(null);
  const activeItem:
    | (ParameterData["layers"][number] & { dragId: number })
    | undefined = useMemo(() => {
    if (!active) return;
    if (active.data.current && "inventory" in active.data.current) {
      return {
        dragId: layerInventoryIdsMap[active.data.current.inventory.value],
        value: active.data.current.inventory.value,
        parameters: STEP_SETTINGS.PARAMETERS.layers[
          active.data.current.inventory
            .value as (typeof STEP_SETTINGS.PARAMETERS.layerValues)[number]
        ].parameters.map(() => ""),
      };
    }
    const index = layerIdsList.findIndex((id) => id === active.id);
    if (index !== -1) {
      return { ...fields[index], ...{ dragId: layerIdsList[index] } };
    }
    return;
  }, [active, fields]);
  const sensors = useSensors(
    useCustomPointerSensor(),
    useCustomKeyboardSensor({ coordinateGetter: sortableKeyboardCoordinates })
  );
  const move = (from: number, to: number) => {
    fieldsMove(from, to);
    setLayerIdsList(arrayMove(layerIdsList, from, to));
  };
  const insert = (
    index: number,
    item: ParameterData["layers"][number] & { dragId: number }
  ) => {
    fieldsInsert(index, item);
    setLayerIdsList([
      ...layerIdsList.slice(0, index),
      item.dragId,
      ...layerIdsList.slice(index),
    ]);
  };
  const remove = (index: number) => {
    fieldsRemove(index);
    setLayerIdsList([
      ...layerIdsList.slice(0, index),
      ...layerIdsList.slice(index + 1),
    ]);
  };
  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={({ active: activeStart }) => {
        if (active !== null) return;
        setActive(activeStart);
      }}
      onDragOver={({ active: activeOver, over }) => {
        if (!activeItem || !over || !activeOver.data.current) return;
        if ("inventory" in activeOver.data.current) {
          insert(
            layerIdsList.findIndex((id) => id === over.id),
            activeItem
          );
        }
      }}
      onDragEnd={({ active: activeEnd, over }) => {
        if (activeEnd.data.current && active && active.data.current) {
          if ("inventory" in activeEnd.data.current && !over) {
            remove(layerIdsList.findIndex((id) => id === activeEnd.id));
          } else if ("sortable" in activeEnd.data.current && over) {
            if ("inventory" in active.data.current) {
              setLayerInventoryIdsMap({
                ...layerInventoryIdsMap,
                [active.data.current.inventory.value]: generateDragId(),
              });
            }
            if (activeEnd.id !== over.id) {
              move(
                layerIdsList.findIndex((id) => id === activeEnd.id),
                layerIdsList.findIndex((id) => id === over.id)
              );
            }
          }
        }
        setActive(null);
      }}
      onDragCancel={() => setActive(null)}
    >
      <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
        <Stack alignItems={"center"} spacing={2} padding={2}>
          <Typography variant="h2" fontSize={25}>
            Layers
          </Typography>
          <Stack direction={"row"} spacing={3}>
            {STEP_SETTINGS.PARAMETERS.layerValues.map((value) => (
              <LayerInventoryComponent
                key={value}
                id={layerInventoryIdsMap[value]}
                value={value}
              />
            ))}
          </Stack>
        </Stack>
      </Paper>
      <SortableContext
        items={layerIdsList}
        strategy={verticalListSortingStrategy}
      >
        <Container>
          <Stack spacing={0}>
            {fields.length > 0 ? (
              fields.map((field, index) => (
                <LayerComponent
                  key={layerIdsList[index]}
                  id={layerIdsList[index]}
                  data={field}
                  layerIndex={index}
                  control={control}
                  remove={() => remove(index)}
                />
              ))
            ) : (
              <Card>This is Unimplemented</Card>
            )}
          </Stack>
        </Container>
      </SortableContext>
      <ClientOnlyPortal selector="#portal">
        <DragOverlay style={{ width: undefined }}>
          {activeItem ? (
            <LayerComponent
              id={activeItem.dragId}
              data={activeItem}
              control={control}
              layerIndex={
                "id" in activeItem
                  ? layerIdsList.findIndex((id) => id === activeItem.dragId)
                  : undefined
              }
              remove={() =>
                remove(layerIdsList.findIndex((id) => id === activeItem.dragId))
              }
            />
          ) : null}
        </DragOverlay>
      </ClientOnlyPortal>
    </DndContext>
  );
};

const LayerInventoryComponent = ({
  id,
  value,
}: {
  id: number;
  value: (typeof STEP_SETTINGS.PARAMETERS.layerValues)[number];
}) => {
  const { attributes, listeners, transform, isDragging, setNodeRef } =
    useDraggable({
      id: id,
      data: {
        inventory: {
          value,
        },
      },
    });
  const style = {
    transform: CSS.Transform.toString(transform),
    display: isDragging ? "none" : undefined,
  };
  return (
    <div ref={setNodeRef} {...attributes} {...listeners}>
      <Card sx={{ p: 1 }} style={{ ...{ display: "inline-block" }, ...style }}>
        {STEP_SETTINGS.PARAMETERS.layers[value].label}
      </Card>
    </div>
  );
};

const LayerComponent = ({
  id,
  data,
  layerIndex,
  control,
  remove,
}: {
  id: number;
  data: ParameterData["layers"][number];
  layerIndex?: number;
  control: Control<ParameterData, unknown>;
  remove: () => void;
}) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    isDragging,
    transform,
    transition,
  } = useSortable({ id });
  const style = {
    opacity: isDragging ? 0.4 : undefined,
    transform: CSS.Transform.toString(transform),
    margin: -2,
    transition: transition,
  };
  return (
    <div ref={setNodeRef}>
      <Card
        sx={{ p: 3 }}
        style={{ ...{ display: "inline-block" }, ...style }}
        {...attributes}
        {...listeners}
      >
        <Stack
          direction={"row"}
          justifyContent={"space-between"}
          alignItems={"center"}
          spacing={3}
        >
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
                    {layerIndex !== undefined ? (
                      <Controller
                        name={`layers.${layerIndex}.parameters.${index}`}
                        control={control}
                        rules={{ required: true }}
                        render={({ field: { onChange, value } }) => (
                          <TextField
                            label={parameter.label}
                            size={"small"}
                            type={parameter.type}
                            onChange={onChange}
                            value={value}
                            required
                          />
                        )}
                      />
                    ) : (
                      <TextField
                        label={parameter.label}
                        size={"small"}
                        type={parameter.type}
                        required
                      />
                    )}
                  </div>
                )
              )}
            </Stack>
            <div data-no-dnd>
              <IconButton onClick={() => remove()}>
                <DeleteIcon />
              </IconButton>
            </div>
          </Stack>
        </Stack>
      </Card>
    </div>
  );
};

export default TabularParametersStep;
