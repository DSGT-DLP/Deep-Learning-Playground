import { useSortable } from "@dnd-kit/sortable";
import { Typography, Stack, Divider, TextField, IconButton, Card } from "@mui/material";
import DeleteIcon from "@mui/icons-material/Delete";
import { CSS } from "@dnd-kit/utilities";
import { Control, FieldErrors, Controller } from "react-hook-form";
import { STEP_SETTINGS } from "../features/Tabular/constants/tabularConstants";
import { ParameterData } from "../features/Tabular/types/tabularTypes";

export const LayerComponent = ({
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