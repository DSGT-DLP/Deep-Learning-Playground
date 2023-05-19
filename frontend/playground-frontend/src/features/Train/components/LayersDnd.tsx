import ClientOnlyPortal from "@/common/components/ClientOnlyPortal";
import { useCustomPointerSensor, useCustomKeyboardSensor } from "@/common/utils/dndHelpers";
import { Active, useSensors, DndContext, closestCenter, DragOverlay } from "@dnd-kit/core";
import { sortableKeyboardCoordinates, SortableContext, verticalListSortingStrategy } from "@dnd-kit/sortable";
import { Stack, Paper, Typography, Container } from "@mui/material";
import { useState, useMemo } from "react";
import { Card } from "react-bootstrap";
import { Control, FieldErrors, useFieldArray } from "react-hook-form";
import { STEP_SETTINGS } from "../features/Tabular/constants/tabularConstants";
import { ParameterData } from "../features/Tabular/types/tabularTypes";

export const LayersDnd = ({
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

// export default LayersDnd;