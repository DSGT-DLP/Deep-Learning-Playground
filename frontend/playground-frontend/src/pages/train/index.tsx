import Footer from "@/common/components/Footer";
import NavbarMain from "@/common/components/NavBarMain";
import {
  Button,
  FormControl,
  FormHelperText,
  Grid,
  InputLabel,
  MenuItem,
  Select,
  TextField,
  Typography,
} from "@mui/material";
import {
  BaseTrainspaceData,
  DATA_SOURCE_ARR,
} from "@/features/Train/types/trainTypes";
import React from "react";
import startCase from "lodash.startcase";
import camelCase from "lodash.camelcase";
import { Controller, useForm } from "react-hook-form";

const TrainSpaceNew = () => {
  const {
    handleSubmit,
    formState: { errors },
    register,
    control,
  } = useForm<BaseTrainspaceData>({
    defaultValues: { name: "My Trainspace" },
  });
  const onSubmit = handleSubmit((data: BaseTrainspaceData) => {
    console.log(data);
  });
  return (
    <div style={{ height: "100vh" }}>
      <NavbarMain />
      <Grid
        container
        spacing={5}
        direction="column"
        alignItems="center"
        justifyContent="center"
        style={{ minHeight: "100vh" }}
      >
        <Grid item>
          <Typography variant="h1" fontSize={50}>
            Create a Trainspace
          </Typography>
        </Grid>
        <Grid item>
          <TextField
            id="filled-basic"
            label="Name"
            variant="filled"
            required
            helperText={errors.name ? "Name is required" : ""}
            error={errors.name ? true : false}
            style={{ width: "500px" }}
            {...register("name", { required: true })}
          />
        </Grid>
        <Grid item>
          <Controller
            name="dataSource"
            control={control}
            rules={{ required: true }}
            render={({ field: { onChange, value } }) => (
              <TextField
                select
                label="Data Source"
                sx={{ m: 1, minWidth: 300 }}
                size="medium"
                required
                defaultValue={""}
                error={errors.dataSource ? true : false}
                onChange={(e) => onChange(e.target.value)}
                value={value ?? ""}
                helperText={errors.dataSource ? "Data Source is required" : ""}
              >
                {DATA_SOURCE_ARR.map((source) => (
                  <MenuItem key={source} value={source}>
                    {source === "CLASSICAL_ML"
                      ? "Classical ML"
                      : startCase(camelCase(source))}
                  </MenuItem>
                ))}
              </TextField>
            )}
          />
        </Grid>
        <Grid item>
          <Button variant="contained" onClick={onSubmit}>
            Create
          </Button>
        </Grid>
      </Grid>
      <Footer />
    </div>
  );
};

export default TrainSpaceNew;
