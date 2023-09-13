import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import {
  Autocomplete,
  Button,
  FormControl,
  FormControlLabel,
  FormGroup,
  FormLabel,
  MenuItem,
  Paper,
  Radio,
  RadioGroup,
  Skeleton,
  Slider,
  Stack,
  Switch,
  TextField,
  Typography,
} from "@mui/material";
import React, { useEffect } from "react";
import { Controller, useForm } from "react-hook-form";
import { DEFAULT_LAYERS, STEP_SETTINGS } from "../constants/tabularConstants";
import { updateTabularTrainspaceData } from "../redux/tabularActions";
import { ParameterData, TrainspaceData } from "../types/tabularTypes";

import { useLazyGetColumnsFromDatasetQuery } from "@/features/Train/redux/trainspaceApi";
import "reactflow/dist/style.css";
import TabularFlow from "./TabularFlow";

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
    setValue,
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
      layers: trainspace?.parameterData?.layers ?? DEFAULT_LAYERS.IRIS,
    },
  });

  useEffect(() => {
    setIsModified(isDirty);
  }, [isDirty]);
  if (!trainspace) return null;
  const targetCol = watch("targetCol");
  const features = watch("features");
  const [layers, setLayers] = React.useState<ParameterData["layers"]>([]);

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
      <Paper>
        <TabularFlow setLayers={setLayers} />
      </Paper>
      {renderStepperButtons((trainspaceData) => {
        setValue("layers", layers);
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

export default TabularParametersStep;
