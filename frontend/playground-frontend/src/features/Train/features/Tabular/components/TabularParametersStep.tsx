import React from "react";
import {
  TabularData,
  TabularParameterData,
} from "@/features/Train/features/Tabular/types/tabularTypes";
import { useGetColumnsFromDatasetQuery } from "@/features/Train/redux/trainspaceApi";
import { useAppSelector } from "@/common/redux/hooks";
import {
  Autocomplete,
  AutocompleteRenderInputParams,
  FormControl,
  FormControlLabel,
  FormLabel,
  Radio,
  RadioGroup,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import {
  Controller,
  ControllerFieldState,
  ControllerRenderProps,
  FieldValues,
  UseFormStateReturn,
  useForm,
} from "react-hook-form";
import camelCase from "lodash.camelcase";
import startCase from "lodash.startcase";
import { TABULAR_PROBLEM_TYPES_ARR } from "../constants/tabularConstants";

const TabularParametersStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TabularData) => void
  ) => React.ReactNode;
}) => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TabularData<"PARAMETERS"> | undefined
  );
  if (!trainspace) return <></>;
  const { data, refetch } = useGetColumnsFromDatasetQuery({
    dataSource: "TABULAR",
    dataset: trainspace.datasetData,
  });
  const {
    handleSubmit,
    formState: { errors },
    register,
    control,
    setValue,
  } = useForm<TabularParameterData>({
    defaultValues: {},
  });
  if (!data) return <></>;

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
          render={({ field: { onChange, value } }) => (
            <RadioGroup row value={value} onChange={onChange}>
              {TABULAR_PROBLEM_TYPES_ARR.map((problemType) => (
                <FormControlLabel
                  key={problemType}
                  value={problemType}
                  control={<Radio />}
                  label={startCase(camelCase(problemType))}
                />
              ))}
            </RadioGroup>
          )}
        />
      </FormControl>
    </Stack>
  );
};

export default TabularParametersStep;
