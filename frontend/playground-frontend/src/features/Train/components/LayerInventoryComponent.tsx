import { useDraggable } from "@dnd-kit/core";
import { Card } from "@mui/material";
// import { STEP_SETTINGS } from "../features/Train/constants/trainConstants";

export const LayerInventoryComponent = ({
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