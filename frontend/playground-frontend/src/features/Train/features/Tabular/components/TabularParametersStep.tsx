import React, { useEffect, useMemo, useState } from "react";
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
import {
  Control,
  Controller,
  FieldArrayWithId,
  useFieldArray,
  useForm,
} from "react-hook-form";
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
  useDroppable,
  useSensors,
} from "@dnd-kit/core";
import DeleteIcon from "@mui/icons-material/Delete";
import {
  WithDndId,
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
  const { fields, move, insert, remove } = useFieldArray({
    control: control,
    name: "layers",
  });
  const genLayerInvIds = () =>
    Object.fromEntries(
      STEP_SETTINGS.PARAMETERS.layerValues.map((layerValue) => [
        layerValue,
        Math.floor(Math.random() * Date.now()),
      ])
    );
  const [layerInvIds, setLayerInvIds] = useState<{
    [layerValue: string]: number;
  }>(genLayerInvIds());
  const [dndActive, setDndActive] = useState<Active | null>(null);
  const [invHovering, setInvHovering] = useState<boolean>(false);

  const dndActiveItem = useMemo(() => {
    if (!dndActive) return;
    if (dndActive.data.current && "inventory" in dndActive.data.current) {
      const value = dndActive.data.current.inventory
        .value as (typeof STEP_SETTINGS.PARAMETERS.layerValues)[number];
      return {
        id: layerInvIds[value],
        value: value,
        parameters: STEP_SETTINGS.PARAMETERS.layers[value].parameters.map(
          () => ""
        ) as ""[],
      };
    } else if (dndActive.data.current && "sortable" in dndActive.data.current) {
      return fields[dndActive.data.current.sortable.index];
    }
  }, [dndActive]);
  const sensors = useSensors(
    useCustomPointerSensor(),
    useCustomKeyboardSensor({ coordinateGetter: sortableKeyboardCoordinates })
  );
  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={({ active }) => {
        if (dndActive !== null) return;
        setDndActive(active);
      }}
      onDragOver={({ over }) => {
        if (!over || !over.data.current) {
          setInvHovering(false);
          return;
        }
        if (!invHovering) {
          setInvHovering(true);
        }
      }}
      onDragEnd={({ active, over }) => {
        if (dndActive && dndActive.data.current && dndActiveItem) {
          if (
            "inventory" in dndActive.data.current &&
            over &&
            over.data.current &&
            "sortable" in over.data.current
          ) {
            insert(over.data.current.sortable.index, {
              value: dndActiveItem.value,
              parameters: dndActiveItem.parameters,
            });
          } else if (
            "sortable" in dndActive.data.current &&
            over &&
            over.data.current &&
            "sortable" in over.data.current
          ) {
            move(
              fields.findIndex((field) => field.id === active.id),
              fields.findIndex((field) => field.id === over.id)
            );
          }
        }
        setLayerInvIds(genLayerInvIds());
        setInvHovering(false);
        setDndActive(null);
      }}
      onDragCancel={({ active }) => {
        if (active.data.current && "inventory" in active.data.current) {
          const index = fields.findIndex((field) => field.id === active.id);
          if (index !== -1) {
            remove(fields.findIndex((field) => field.id === active.id));
          }
        }
        setLayerInvIds(genLayerInvIds());
        setInvHovering(false);
        setDndActive(null);
      }}
    >
      <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
        <Stack alignItems={"center"} spacing={2} padding={2}>
          <Typography variant="h2" fontSize={25}>
            Layers
          </Typography>
          <Stack direction={"row"} spacing={3}>
            {STEP_SETTINGS.PARAMETERS.layerValues.map((value) => (
              <LayerInventoryComponent
                id={layerInvIds[value]}
                key={value}
                value={value}
              />
            ))}
          </Stack>
        </Stack>
      </Paper>
      <SortableContext
        items={
          dndActiveItem &&
          dndActive?.data.current &&
          "inventory" in dndActive.data.current
            ? [dndActiveItem, ...fields]
            : fields
        }
        strategy={verticalListSortingStrategy}
      >
        <Container>
          <Stack spacing={0}>
            {fields.length > 0 ? (
              [
                dndActiveItem &&
                dndActive?.data.current &&
                "inventory" in dndActive.data.current &&
                invHovering ? (
                  <LayerComponent
                    key={dndActiveItem.id}
                    id={dndActiveItem.id}
                    data={dndActiveItem}
                  />
                ) : null,
                ...fields.map((field, index) => (
                  <LayerComponent
                    key={field.id}
                    id={field.id}
                    data={field}
                    layerProps={{
                      index: index,
                      control: control,
                      remove: () => remove(index),
                    }}
                  />
                )),
              ]
            ) : (
              <Card>This is Unimplemented</Card>
            )}
          </Stack>
        </Container>
      </SortableContext>
      <ClientOnlyPortal selector="#portal">
        <DragOverlay style={{ width: undefined }}>
          {dndActiveItem ? <LayerComponent data={dndActiveItem} /> : null}
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
  const { attributes, listeners, isDragging, setNodeRef } = useDraggable({
    id: id,
    data: {
      inventory: {
        value,
      },
    },
  });

  const style = {
    opacity: isDragging ? 0.4 : undefined,
  };
  return (
    <div ref={setNodeRef}>
      <Card
        sx={{ p: 1 }}
        style={{ ...{ display: "inline-block" }, ...style }}
        {...attributes}
        {...listeners}
      >
        {STEP_SETTINGS.PARAMETERS.layers[value].label}
      </Card>
    </div>
  );
};

const LayerComponent = ({
  id,
  data,
  layerProps,
}: {
  id?: string | number;
  data: ParameterData["layers"][number];
  layerProps?: {
    index: number;
    control: Control<ParameterData, unknown>;
    remove: () => void;
  };
}) => {
  const {
    attributes,
    listeners,
    setNodeRef,
    isDragging,
    transform,
    transition,
  } = id
    ? useSortable({ id })
    : {
        attributes: undefined,
        listeners: undefined,
        setNodeRef: undefined,
        isDragging: undefined,
        transform: undefined,
        transition: undefined,
      };
  const style = transform
    ? {
        opacity: isDragging ? 0.4 : undefined,
        transform: CSS.Transform.toString(transform),
        transition: transition,
      }
    : undefined;
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
                    {layerProps ? (
                      <Controller
                        name={`layers.${layerProps.index}.parameters.${index}`}
                        control={layerProps.control}
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
                        value={""}
                        required
                      />
                    )}
                  </div>
                )
              )}
            </Stack>
            <div data-no-dnd>
              <IconButton onClick={layerProps?.remove}>
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
