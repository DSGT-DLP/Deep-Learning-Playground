import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import InfoIcon from "@mui/icons-material/Info";
import Tooltip, { tooltipClasses } from "@mui/material/Tooltip";
import { styled } from "@mui/material/styles";
import React, { useEffect, useMemo, useState } from "react";

import ClientOnlyPortal from "@/common/components/ClientOnlyPortal";
import {
  useCustomKeyboardSensor,
  useCustomPointerSensor,
} from "@/common/utils/dndHelpers";
import {
  Active,
  DndContext,
  DragOverlay,
  closestCenter,
  useDraggable,
  useSensors,
} from "@dnd-kit/core";
import {
  SortableContext,
  sortableKeyboardCoordinates,
  useSortable,
  verticalListSortingStrategy,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import DeleteIcon from "@mui/icons-material/Delete";
import {
  Card,
  Container,
  Divider,
  FormControlLabel,
  FormGroup,
  IconButton,
  MenuItem,
  Paper,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import {
  Control,
  Controller,
  FieldErrors,
  useFieldArray,
  useForm,
} from "react-hook-form";
import { STEP_SETTINGS } from "../constants/imageConstants";
import { updateImageTrainspaceData } from "../redux/imageActions";
import { ParameterData, TrainspaceData } from "../types/imageTypes";

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

const ImageParametersStep = ({
  renderStepperButtons,
  setIsModified,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"PARAMETERS">) => void
  ) => React.ReactNode;
  setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const trainspace = useAppSelector(
    (state) =>
      state.trainspace.current as TrainspaceData<"PARAMETERS"> | undefined
  );
  const dispatch = useAppDispatch();
  const {
    handleSubmit,
    formState: { errors, isDirty },
    control,
  } = useForm<ParameterData>({
    defaultValues: {
      criterion: trainspace?.parameterData?.criterion ?? "CELOSS",
      optimizerName: trainspace?.parameterData?.optimizerName ?? "SGD",
      shuffle: trainspace?.parameterData?.shuffle ?? true,
      epochs: trainspace?.parameterData?.epochs ?? 5,
      batchSize: trainspace?.parameterData?.batchSize ?? 20,
      trainTransforms: trainspace?.parameterData?.trainTransforms ?? [
        {
          value: "GRAYSCALE",
          parameters: [],
        },
        {
          value: "TO_TENSOR",
          parameters: [],
        },
        {
          value: "RESIZE",
          parameters: [32, 32],
        },
      ],
      testTransforms: trainspace?.parameterData?.testTransforms ?? [
        {
          value: "GRAYSCALE",
          parameters: [],
        },
        {
          value: "TO_TENSOR",
          parameters: [],
        },
        {
          value: "RESIZE",
          parameters: [32, 32],
        },
      ],
      layers: trainspace?.parameterData?.layers ?? [
        {
          value: "CONV2D",
          parameters: [1, 5, 3, 1, 1],
        },
        {
          value: "MAXPOOL2D",
          parameters: [3, 1],
        },
        {
          value: "FLATTEN",
          parameters: [1, -1],
        },
        {
          value: "LINEAR",
          parameters: [500, 10],
        },
        {
          value: "SIGMOID",
          parameters: [],
        },
      ],
    },
  });
  useEffect(() => {
    setIsModified(isDirty);
  }, [isDirty]);
  if (!trainspace) return <></>;
  return (
    <Stack spacing={3}>
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
      <TrainTransformsDnd control={control} errors={errors} />
      <TestTransformsDnd control={control} errors={errors} />
      <LayersDnd control={control} errors={errors} />
      {renderStepperButtons((trainspaceData) => {
        handleSubmit((data) => {
          dispatch(
            updateImageTrainspaceData({
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
                            style={{ minWidth: "124px" }}
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

const TrainTransformsDnd = ({
  control,
  errors,
}: {
  control: Control<ParameterData, unknown>;
  errors: FieldErrors<ParameterData>;
}) => {
  const { fields, move, insert, remove } = useFieldArray({
    control: control,
    name: "trainTransforms",
  });
  const genTransformInvIds = () =>
    Object.fromEntries(
      STEP_SETTINGS.PARAMETERS.transformValues.map((transformValue) => [
        transformValue,
        Math.floor(Math.random() * Date.now()),
      ])
    );
  const [transformInvIds, setTransformInvIds] = useState<{
    [transformValue: string]: number;
  }>(genTransformInvIds());
  const [dndActive, setDndActive] = useState<Active | null>(null);
  const [invHovering, setInvHovering] = useState<boolean>(false);

  const dndActiveItem = useMemo(() => {
    if (!dndActive) return;
    if (dndActive.data.current && "inventory" in dndActive.data.current) {
      const value = dndActive.data.current.inventory
        .value as (typeof STEP_SETTINGS.PARAMETERS.transformValues)[number];
      return {
        id: transformInvIds[value],
        value: value,
        parameters: STEP_SETTINGS.PARAMETERS.transforms[value].parameters.map(
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
        setTransformInvIds(genTransformInvIds());
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
        setTransformInvIds(genTransformInvIds());
        setInvHovering(false);
        setDndActive(null);
      }}
    >
      <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
        <Stack alignItems={"center"} spacing={2} padding={2}>
          <Typography variant="h2" fontSize={25}>
            Train Transforms
          </Typography>
          <Stack
            direction={"row"}
            spacing={2}
            justifyContent={"center"}
            sx={{ flexWrap: "wrap", gap: 1 }}
          >
            {STEP_SETTINGS.PARAMETERS.transformValues.map((value) => (
              <TrainTransformInventoryComponent
                id={transformInvIds[value]}
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
                  <TrainTransformComponent
                    key={dndActiveItem.id}
                    id={dndActiveItem.id}
                    data={
                      dndActiveItem as ParameterData["trainTransforms"][number]
                    }
                  />
                ) : null,
                ...fields.map((field, index) => (
                  <TrainTransformComponent
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
              <TrainTransformComponent
                data={dndActiveItem as ParameterData["trainTransforms"][number]}
                formProps={{
                  index: dndActive.data.current.sortable.index,
                  control: control,
                  errors: errors,
                }}
              />
            ) : (
              <TrainTransformComponent
                data={dndActiveItem as ParameterData["trainTransforms"][number]}
              />
            )
          ) : null}
        </DragOverlay>
      </ClientOnlyPortal>
    </DndContext>
  );
};

const TrainTransformComponent = ({
  id,
  data,
  formProps,
}: {
  id?: string | number;
  data: ParameterData["trainTransforms"][number];
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
          <Typography variant="h3" fontSize={18}>
            {STEP_SETTINGS.PARAMETERS.transforms[data.value].label}
          </Typography>
          <Stack direction={"row"} alignItems={"center"} spacing={3}>
            <Stack
              direction={"row"}
              alignItems={"center"}
              justifyContent={"flex-end"}
              spacing={2}
              divider={<Divider orientation="vertical" flexItem />}
            >
              {STEP_SETTINGS.PARAMETERS.transforms[data.value].parameters.map(
                (parameter, index) => (
                  <div key={index} data-no-dnd>
                    {formProps ? (
                      <Controller
                        name={`trainTransforms.${formProps.index}.parameters.${index}`}
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
                              formProps.errors.trainTransforms?.[
                                formProps.index
                              ]?.parameters?.[index]
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

const TrainTransformInventoryComponent = ({
  id,
  value,
}: {
  id: number;
  value: (typeof STEP_SETTINGS.PARAMETERS.transformValues)[number];
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
        {STEP_SETTINGS.PARAMETERS.transforms[value].label}
      </Card>
    </div>
  );
};

const TestTransformsDnd = ({
  control,
  errors,
}: {
  control: Control<ParameterData, unknown>;
  errors: FieldErrors<ParameterData>;
}) => {
  const { fields, move, insert, remove } = useFieldArray({
    control: control,
    name: "testTransforms",
  });
  const genTransformInvIds = () =>
    Object.fromEntries(
      STEP_SETTINGS.PARAMETERS.transformValues.map((transformValue) => [
        transformValue,
        Math.floor(Math.random() * Date.now()),
      ])
    );
  const [transformInvIds, setTransformInvIds] = useState<{
    [transformValue: string]: number;
  }>(genTransformInvIds());
  const [dndActive, setDndActive] = useState<Active | null>(null);
  const [invHovering, setInvHovering] = useState<boolean>(false);

  const dndActiveItem = useMemo(() => {
    if (!dndActive) return;
    if (dndActive.data.current && "inventory" in dndActive.data.current) {
      const value = dndActive.data.current.inventory
        .value as (typeof STEP_SETTINGS.PARAMETERS.transformValues)[number];
      return {
        id: transformInvIds[value],
        value: value,
        parameters: STEP_SETTINGS.PARAMETERS.transforms[value].parameters.map(
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
        setTransformInvIds(genTransformInvIds());
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
        setTransformInvIds(genTransformInvIds());
        setInvHovering(false);
        setDndActive(null);
      }}
    >
      <Paper elevation={1} style={{ backgroundColor: "transparent" }}>
        <Stack alignItems={"center"} spacing={2} padding={2}>
          <Typography variant="h2" fontSize={25}>
            Test Transforms
          </Typography>
          <Stack
            direction={"row"}
            spacing={2}
            justifyContent={"center"}
            sx={{ flexWrap: "wrap", gap: 1 }}
          >
            {STEP_SETTINGS.PARAMETERS.transformValues.map((value) => (
              <TestTransformInventoryComponent
                id={transformInvIds[value]}
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
                  <TestTransformComponent
                    key={dndActiveItem.id}
                    id={dndActiveItem.id}
                    data={
                      dndActiveItem as ParameterData["testTransforms"][number]
                    }
                  />
                ) : null,
                ...fields.map((field, index) => (
                  <TestTransformComponent
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
              <TestTransformComponent
                data={dndActiveItem as ParameterData["testTransforms"][number]}
                formProps={{
                  index: dndActive.data.current.sortable.index,
                  control: control,
                  errors: errors,
                }}
              />
            ) : (
              <TestTransformComponent
                data={dndActiveItem as ParameterData["testTransforms"][number]}
              />
            )
          ) : null}
        </DragOverlay>
      </ClientOnlyPortal>
    </DndContext>
  );
};

const TestTransformComponent = ({
  id,
  data,
  formProps,
}: {
  id?: string | number;
  data: ParameterData["testTransforms"][number];
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
          <Typography variant="h3" fontSize={18}>
            {STEP_SETTINGS.PARAMETERS.transforms[data.value].label}
          </Typography>
          <Stack direction={"row"} alignItems={"center"} spacing={3}>
            <Stack
              direction={"row"}
              alignItems={"center"}
              justifyContent={"flex-end"}
              spacing={2}
              divider={<Divider orientation="vertical" flexItem />}
            >
              {STEP_SETTINGS.PARAMETERS.transforms[data.value].parameters.map(
                (parameter, index) => (
                  <div key={index} data-no-dnd>
                    {formProps ? (
                      <Controller
                        name={`testTransforms.${formProps.index}.parameters.${index}`}
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
                              formProps.errors.testTransforms?.[formProps.index]
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

const TestTransformInventoryComponent = ({
  id,
  value,
}: {
  id: number;
  value: (typeof STEP_SETTINGS.PARAMETERS.transformValues)[number];
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
        {STEP_SETTINGS.PARAMETERS.transforms[value].label}
      </Card>
    </div>
  );
};

export default ImageParametersStep;
