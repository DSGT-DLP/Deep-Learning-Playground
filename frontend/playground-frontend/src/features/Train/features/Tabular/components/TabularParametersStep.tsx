import React, { useEffect, useState } from "react";
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
  DndContext,
  DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";

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
          parameters: [
            {
              data: 10,
              value: "INPUT_SIZE",
            },
          ],
        },
        {
          value: "RELU",
          parameters: [],
        },
        {
          value: "LINEAR",
          parameters: [
            {
              data: 10,
              value: "INPUT_SIZE",
            },
          ],
        },
        {
          value: "SOFTMAX",
          parameters: [
            {
              value: "DIMENSION",
              data: -1,
            },
          ],
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

const DragAndDropList = ({
  control,
}: {
  control: Control<ParameterData, unknown>;
}) => {
  const { fields, move } = useFieldArray({
    control: control,
    name: "layers",
  });
  const sensors = useSensors(
    useSensor(PointerSensor),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    })
  );
  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      console.log(active.id, over.id);

      move(
        fields.findIndex((item) => item.id === active.id),
        fields.findIndex((item) => item.id === over.id)
      );
    }
  };
  return (
    <DndContext
      sensors={sensors}
      collisionDetection={closestCenter}
      onDragEnd={handleDragEnd}
    >
      <SortableContext items={fields} strategy={verticalListSortingStrategy}>
        {fields.map((field) => (
          <SortableItem key={field.id} id={field.id} value={field.value} />
        ))}
      </SortableContext>
    </DndContext>
  );
};

const SortableItem = ({ id, value }: { id: string; value: string }) => {
  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition: transition,
  };

  return (
    <Card ref={setNodeRef} style={style} {...attributes} {...listeners}>
      {value}
    </Card>
  );
};

export default TabularParametersStep;
