import React, { useEffect, useMemo, useState } from "react";
import { useLazyGetColumnsFromDatasetQuery } from "@/features/Train/redux/trainspaceApi";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
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
  Skeleton,
  Button,
} from "@mui/material";
import { styled } from "@mui/material/styles";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import InfoIcon from "@mui/icons-material/Info";
import {
  Control,
  Controller,
  FieldErrors,
  useFieldArray,
  useForm,
} from "react-hook-form";
import { ParameterData, TrainspaceData } from "../types/tabularTypes";
import { STEP_SETTINGS } from "../constants/tabularConstants";
import { CSS } from "@dnd-kit/utilities";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import {
  Active,
  DndContext,
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
import { updateTabularTrainspaceData } from "../redux/tabularActions";

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

const TabularParametersStep = ({
  renderStepperButtons,
  setIsModified,
  setStep,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"PARAMETERS">) => void
  ) => React.ReactNode;
  setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
  setStep: React.Dispatch<React.SetStateAction<number>>;
}) => {
  const trainspace = useAppSelector(
    (state) =>
      state.trainspace.current as TrainspaceData<"PARAMETERS"> | undefined
  );
  const dispatch = useAppDispatch();
  const [getColumns, { data, error }] = useLazyGetColumnsFromDatasetQuery();
  useEffect(() => {
    trainspace &&
      getColumns({
        dataSource: trainspace.dataSource,
        dataset: trainspace.datasetData,
      });
  }, []);
  const {
    handleSubmit,
    formState: { errors, isDirty },
    control,
    watch,
  } = useForm<ParameterData>({
    defaultValues: {
      targetCol:
        trainspace?.parameterData?.targetCol ?? (null as unknown as undefined),
      features: trainspace?.parameterData?.features ?? [],
      problemType: trainspace?.parameterData?.problemType ?? "CLASSIFICATION",
      criterion: trainspace?.parameterData?.criterion ?? "CELOSS",
      optimizerName: trainspace?.parameterData?.optimizerName ?? "SGD",
      shuffle: trainspace?.parameterData?.shuffle ?? true,
      epochs: trainspace?.parameterData?.epochs ?? 5,
      batchSize: trainspace?.parameterData?.batchSize ?? 20,
      testSize: trainspace?.parameterData?.testSize ?? 0.2,
      layers: trainspace?.parameterData?.layers ?? [
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

  useEffect(() => {
    setIsModified(isDirty);
  }, [isDirty]);
  if (!trainspace) return <></>;
  const targetCol = watch("targetCol");
  const features = watch("features");
  return (
    <Stack spacing={3}>
      {error ? (
        <>
          <Typography variant="h2" fontSize={25} textAlign={"center"}>
            Error Occurred!
          </Typography>
          <Button
            onClick={() =>
              getColumns({
                dataSource: "TABULAR",
                dataset: trainspace.datasetData,
              })
            }
          >
            Retry
          </Button>
          <Button onClick={() => setStep(0)}>Reupload</Button>
        </>
      ) : (
        <>
          <Controller
            control={control}
            name="targetCol"
            rules={{ required: true }}
            render={({ field: { ref, onChange, ...field } }) => (
              <Autocomplete
                {...field}
                onChange={(_, value) => onChange(value)}
                disableClearable
                autoComplete
                autoHighlight
                renderInput={(params) =>
                  data ? (
                    <TextField
                      {...params}
                      inputRef={ref}
                      required
                      label="Target Column"
                      placeholder="Target"
                      error={errors.targetCol ? true : false}
                    />
                  ) : (
                    <Skeleton width="100%">
                      <TextField fullWidth />
                    </Skeleton>
                  )
                }
                options={
                  data ? data.filter((col) => !features.includes(col)) : []
                }
              />
            )}
          />
          <Controller
            control={control}
            name="features"
            rules={{ required: true }}
            render={({ field: { ref, onChange, ...field } }) => (
              <Autocomplete
                {...field}
                multiple
                autoComplete
                autoHighlight
                disableCloseOnSelect
                onChange={(_, value) => onChange(value)}
                renderInput={(params) =>
                  data ? (
                    <TextField
                      {...params}
                      inputRef={ref}
                      required
                      label="Feature Columns"
                      placeholder="Features"
                      error={errors.features ? true : false}
                    />
                  ) : (
                    <Skeleton width="100%">
                      <TextField fullWidth />
                    </Skeleton>
                  )
                }
                options={data ? data.filter((col) => col !== targetCol) : []}
              />
            )}
          />
        </>
      )}
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
      <LayersDnd control={control} errors={errors} />
      {renderStepperButtons((trainspaceData) => {
        handleSubmit((data) => {
          dispatch(
            updateTabularTrainspaceData({
              current: {
                ...trainspaceData,
                parameterData: data,
                reviewData: undefined,
              },
              stepLabel: "PARAMETERS",
            })
          );
        })();
      })}
    </Stack>
  );
};

const LayersDnd = ({
  control,
  errors,
}: {
  control: Control<ParameterData, unknown>;
  errors: FieldErrors<ParameterData>;
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
            over?.data.current &&
            "sortable" in over.data.current
          ) {
            insert(over.data.current.sortable.index, {
              value: dndActiveItem.value,
              parameters: dndActiveItem.parameters as number[],
            });
          } else if (
            "sortable" in dndActive.data.current &&
            over?.data.current &&
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

      <Container>
        <Stack spacing={0}>
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
            {fields.length > 0 ? (
              [
                dndActiveItem &&
                dndActive?.data.current &&
                "inventory" in dndActive.data.current &&
                invHovering ? (
                  <LayerComponent
                    key={dndActiveItem.id}
                    id={dndActiveItem.id}
                    data={dndActiveItem as ParameterData["layers"][number]}
                  />
                ) : null,
                ...fields.map((field, index) => (
                  <LayerComponent
                    key={field.id}
                    id={field.id}
                    data={field}
                    formProps={{
                      index: index,
                      control: control,
                      errors: errors,
                      remove: () => remove(index),
                    }}
                  />
                )),
              ]
            ) : (
              <Card>This is Unimplemented</Card>
            )}
          </SortableContext>
        </Stack>
      </Container>

      <ClientOnlyPortal selector="#portal">
        <DragOverlay style={{ width: undefined }}>
          {dndActiveItem ? (
            dndActive?.data.current && "sortable" in dndActive.data.current ? (
              <LayerComponent
                data={dndActiveItem as ParameterData["layers"][number]}
                formProps={{
                  index: dndActive.data.current.sortable.index,
                  control: control,
                  errors: errors,
                }}
              />
            ) : (
              <LayerComponent
                data={dndActiveItem as ParameterData["layers"][number]}
              />
            )
          ) : null}
        </DragOverlay>
      </ClientOnlyPortal>
    </DndContext>
  );
};

const LayerComponent = ({
  id,
  data,
  formProps,
}: {
  id?: string | number;
  data: ParameterData["layers"][number];
  formProps?: {
    index: number;
    control: Control<ParameterData, unknown>;
    errors: FieldErrors<ParameterData>;
    remove?: () => void;
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

export default TabularParametersStep;
