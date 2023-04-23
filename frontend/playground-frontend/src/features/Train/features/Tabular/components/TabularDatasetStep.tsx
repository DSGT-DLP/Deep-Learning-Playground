import React from "react";
import { useForm } from "react-hook-form";
import { DatasetData } from "@/features/Train/types/trainTypes";
import { useAppDispatch } from "@/common/redux/hooks";
import { setTrainspaceData } from "@/features/Train/redux/trainspaceSlice";
import {
  DatasetStepTabLayout,
  DefaultDatasetPanel,
  UploadDatasetPanel,
} from "@/features/Train/components/DatasetStepLayout";
import { TabularData } from "../types/tabularTypes";

const TabularDatasetStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TabularData) => void
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
            trainspaceData.datasetData = data;
            trainspaceData.step = "PARAMETERS";
            dispatch(setTrainspaceData(trainspaceData));
          })();
        } else {
          defaultDatasetMethods.handleSubmit((data) => {
            trainspaceData.datasetData = data;
            trainspaceData.step = "PARAMETERS";
            dispatch(setTrainspaceData(trainspaceData));
          })();
        }
      })}
    />
  );
};

export default TabularDatasetStep;
