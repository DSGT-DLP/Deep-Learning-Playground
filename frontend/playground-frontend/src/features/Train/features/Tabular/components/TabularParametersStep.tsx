import React, { useEffect, useMemo, useState } from "react";
import { useGetColumnsFromDatasetQuery } from "@/features/Train/redux/trainspaceApi";
import { useAppSelector } from "@/common/redux/hooks";
import {
  Autocomplete,
  Button,
  Card,
  Container,
  Divider,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormLabel,
  IconButton,
  MenuItem,
  Radio,
  RadioGroup,
  Select,
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
import AddIcon from "@mui/icons-material/Add";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import {
  Active,
  DndContext,
  DragEndEvent,
  DragOverlay,
  DraggableAttributes,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { SyntheticListenerMap } from "@dnd-kit/core/dist/hooks/utilities";

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
      <Stack spacing={0}>
        <DragAndDropList control={control} />
      </Stack>
    </Stack>
  );
};

function shouldHandleEvent(element: HTMLElement | null) {
  let cur = element;

  while (cur) {
    if (cur.dataset && cur.dataset.noDnd) {
      return false;
    }
    cur = cur.parentElement;
  }

  return true;
}

export class CustomPointerSensor extends PointerSensor {
  static activators = [
    {
      eventName: "onPointerDown" as const,
      handler: ({ nativeEvent: event }: React.MouseEvent) => {
        return shouldHandleEvent(event.target as HTMLElement);
      },
    },
  ];
}

export class CustomKeyboardSensor extends KeyboardSensor {
  static activators = [
    {
      eventName: "onKeyDown" as const,
      handler: ({ nativeEvent: event }: React.KeyboardEvent<Element>) => {
        return shouldHandleEvent(event.target as HTMLElement);
      },
    },
  ];
}

const DragAndDropList = ({
  control,
}: {
  control: Control<ParameterData, unknown>;
}) => {
  const { fields, move } = useFieldArray({
    control: control,
    name: "layers",
  });
  const [active, setActive] = useState<Active | null>(null);
  const activeItem = useMemo(
    () => fields.find((field) => field.id === active?.id),
    [active, fields]
  );
  const sensors = useSensors(
    useSensor(CustomPointerSensor),
    useSensor(CustomKeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      move(
        fields.findIndex((item) => item.id === active.id),
        fields.findIndex((item) => item.id === over.id)
      );
      setActive(null);
    }
  };
  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragStart={({ active }) => {
        setActive(active);
      }}
      onDragEnd={handleDragEnd}
      onDragCancel={() => setActive(null)}
    >
      <SortableContext items={fields} strategy={verticalListSortingStrategy}>
        {fields.map((field, index) => (
          <SortableItem
            key={field.id}
            id={field.id}
            data={field}
            layerIndex={index}
            control={control}
          />
        ))}
      </SortableContext>
      <DragOverlay>
        {activeItem ? (
          <LayerComponent
            data={activeItem}
            control={control}
            layerIndex={fields.findIndex((item) => item.id === activeItem.id)}
          />
        ) : null}
      </DragOverlay>
    </DndContext>
  );
};

const LayerComponent = ({
  data,
  layerIndex,
  control,
  attributes,
  listeners,
}: {
  data: ParameterData["layers"][number];
  layerIndex: number;
  attributes?: DraggableAttributes;
  listeners?: SyntheticListenerMap;
  control: Control<ParameterData, unknown>;
}) => {
  return (
    <Card
      sx={{ p: 3 }}
      style={{ display: "inline-block" }}
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
        <Stack
          direction={"row"}
          justifyContent={"flex-end"}
          spacing={2}
          divider={<Divider orientation="vertical" flexItem />}
        >
          {STEP_SETTINGS.PARAMETERS.layers[data.value].parameters.map(
            (parameter, index) => (
              <div key={index} data-no-dnd>
                <Controller
                  name={`layers.${layerIndex}.parameters.${index}`}
                  control={control}
                  render={({ field: { onChange, value } }) => (
                    <TextField
                      label={parameter.label}
                      size={"small"}
                      type={parameter.type}
                      onChange={onChange}
                      value={value}
                    />
                  )}
                />
              </div>
            )
          )}
        </Stack>
      </Stack>
    </Card>
  );
};

const SortableItem = ({
  id,
  layerIndex,
  data,
  control,
}: {
  id: string;
  layerIndex: number;
  data: ParameterData["layers"][number];
  control: Control<ParameterData, unknown>;
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
    margin: -2,
    transform: CSS.Transform.toString(transform),
    transition: transition,
  };

  return (
    <div ref={setNodeRef} style={style}>
      <LayerComponent
        data={data}
        layerIndex={layerIndex}
        control={control}
        attributes={attributes}
        listeners={listeners}
      />
    </div>
  );
};

export default TabularParametersStep;
