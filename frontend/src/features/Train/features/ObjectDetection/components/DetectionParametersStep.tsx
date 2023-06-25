import React, { useEffect, useMemo, useState } from "react";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import {
  Card,
  Container,
  Divider,
  FormControl,
  FormControlLabel,
  FormLabel,
  IconButton,
  Paper,
  Radio,
  RadioGroup,
  Stack,
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
import { ParameterData, TrainspaceData } from "../types/detectionTypes";
import { STEP_SETTINGS } from "../constants/detectionConstants";
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
import { updateDetectionTrainspaceData } from "../redux/detectionActions";
import { id } from "date-fns/locale";

const DetectionParametersStep = ({
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
      detectionType: trainspace?.parameterData?.detectionType ?? "rekognition",
      detectionProblemType:
        trainspace?.parameterData?.detectionProblemType ?? "labels",
      transforms: trainspace?.parameterData?.transforms ?? [],
    },
  });
  useEffect(() => {
    setIsModified(isDirty);
  }, [isDirty]);
  if (!trainspace) return <></>;
  return (
    <Stack spacing={3}>
      <FormControl>
        <FormLabel>Detection Type</FormLabel>
        <Controller
          name="detectionType"
          control={control}
          rules={{ required: true }}
          render={({ field: { onChange, value } }) => (
            <RadioGroup row value={value} onChange={onChange}>
              {STEP_SETTINGS["PARAMETERS"].detectionTypes.map(
                (detectionType) => (
                  <FormControlLabel
                    key={detectionType.value}
                    value={detectionType.value}
                    control={<Radio />}
                    label={detectionType.label}
                  />
                )
              )}
            </RadioGroup>
          )}
        />
      </FormControl>
      <FormControl>
        <FormLabel>Detection Problem Type</FormLabel>
        <Controller
          name="detectionProblemType"
          control={control}
          rules={{ required: true }}
          render={({ field: { onChange, value } }) => (
            <RadioGroup row value={value} onChange={onChange}>
              {STEP_SETTINGS["PARAMETERS"].detectionProblemTypes.map(
                (detectionProblemType) => (
                  <FormControlLabel
                    key={detectionProblemType.value}
                    value={detectionProblemType.value}
                    control={<Radio />}
                    label={detectionProblemType.label}
                  />
                )
              )}
            </RadioGroup>
          )}
        />
      </FormControl>
      <LayersDnd control={control} errors={errors} />
      {renderStepperButtons((trainspaceData) => {
        handleSubmit((data) => {
          dispatch(
            updateDetectionTrainspaceData({
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
    name: "transforms",
  });
  const genLayerInvIds = () =>
    Object.fromEntries(
      STEP_SETTINGS.PARAMETERS.detectionTransformValues.map(
        (detectionTransform) => [
          detectionTransform,
          Math.floor(Math.random() * Date.now()),
        ]
      )
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
        .value as (typeof STEP_SETTINGS.PARAMETERS.detectionTransformValues)[number];
      return {
        id: layerInvIds[value],
        value: value,
        parameters: STEP_SETTINGS.PARAMETERS.detectionTransforms[
          value
        ].parameters.map(() => "") as ""[],
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
            {STEP_SETTINGS.PARAMETERS.detectionTransformValues.map((value) => (
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
                    data={dndActiveItem as ParameterData["transforms"][number]}
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
                data={dndActiveItem as ParameterData["transforms"][number]}
                formProps={{
                  index: dndActive.data.current.sortable.index,
                  control: control,
                  errors: errors,
                }}
              />
            ) : (
              <LayerComponent
                data={dndActiveItem as ParameterData["transforms"][number]}
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
  data: ParameterData["transforms"][number];
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
            {
              STEP_SETTINGS.PARAMETERS.detectionTransforms[data.value]
                .display_name
            }
          </Typography>
          <Stack direction={"row"} alignItems={"center"} spacing={3}>
            <Stack
              direction={"row"}
              alignItems={"center"}
              justifyContent={"flex-end"}
              spacing={2}
              divider={<Divider orientation="vertical" flexItem />}
            >
              {STEP_SETTINGS.PARAMETERS.detectionTransforms[
                data.value
              ].parameters.map((parameter, index) => (
                <div key={index} data-no-dnd>
                  {formProps ? (
                    <Controller
                      name={`transforms.${formProps.index}.parameters.${index}`}
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
                            formProps.errors.transforms?.[formProps.index]
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
              ))}
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
  value: (typeof STEP_SETTINGS.PARAMETERS.detectionTransformValues)[number];
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
        {STEP_SETTINGS.PARAMETERS.detectionTransforms[value].label}
      </Card>
    </div>
  );
};

export default DetectionParametersStep;
