import React from "react";
import { useForm } from "react-hook-form";
import { DatasetData } from "@/features/Train/types/trainTypes";
import { useAppDispatch } from "@/common/redux/hooks";
import {
  DatasetStepTabLayout,
  DefaultDatasetPanel,
  UploadDatasetPanel,
} from "@/features/Train/components/DatasetStepLayout";
import { TrainspaceData } from "../types/tabularTypes";
import { updateTabularTrainspaceData } from "../redux/tabularActions";

const TabularDatasetStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"DATASET">) => void
  ) => React.ReactNode;
}) => {
  const [currTab, setCurrTab] = React.useState("upload-dataset");
  const defaultDatasetMethods = useForm<DatasetData>();
  const uploadDatasetMethods = useForm<DatasetData>();
  const dispatch = useAppDispatch();
  return (
    <DatasetStepTabLayout
      currTab={currTab}
      setCurrTab={setCurrTab}
      tabs={[
        { tabLabel: "Recently Uploaded Datasets", tabValue: "upload-dataset" },
        { tabLabel: "Default Datasets", tabValue: "default-dataset" },
      ]}
      tabComponents={{
        "upload-dataset": (
          <UploadDatasetPanel
            dataSource={"TABULAR"}
            methods={uploadDatasetMethods}
          />
        ),
        "default-dataset": (
          <DefaultDatasetPanel methods={defaultDatasetMethods} />
        ),
      }}
      stepperButtons={renderStepperButtons((trainspaceData) => {
        if (currTab === "upload-dataset") {
          uploadDatasetMethods.handleSubmit((data) => {
            dispatch(
              updateTabularTrainspaceData({
                current: {
                  ...trainspaceData,
                  datasetData: data,
                },
                stepLabel: "DATASET",
              })
            );
          })();
        } else {
          defaultDatasetMethods.handleSubmit((data) => {
            dispatch(
              updateTabularTrainspaceData({
                current: {
                  ...trainspaceData,
                  datasetData: data,
                },
                stepLabel: "DATASET",
              })
            );
          })();
        }
      })}
    />
  );
};

export default TabularDatasetStep;
