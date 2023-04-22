import React from "react";
import { useForm } from "react-hook-form";
import {
  BaseTrainspaceData,
  DefaultDatasetData,
  FileDatasetData,
} from "@/features/Train/types/trainTypes";
import { useAppDispatch } from "@/common/redux/hooks";
import { setTrainspaceDataset } from "@/features/Train/constants/trainConstants";
import { setTrainspaceData } from "@/features/Train/redux/trainspaceSlice";
import {
  DatasetStepTabLayout,
  DefaultDatasetPanel,
  UploadDatasetPanel,
} from "@/features/Train/components/DatasetStepLayout";

const TabularDatasetStep = ({
  renderStepperButtons,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: BaseTrainspaceData) => void
  ) => React.ReactNode;
}) => {
  const [currTab, setCurrTab] = React.useState("upload-dataset");
  const defaultDatasetMethods = useForm<DefaultDatasetData>();
  const uploadDatasetMethods = useForm<FileDatasetData>();
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
        "upload-dataset": <UploadDatasetPanel methods={uploadDatasetMethods} />,
        "default-dataset": (
          <DefaultDatasetPanel methods={defaultDatasetMethods} />
        ),
      }}
      stepperButtons={renderStepperButtons((trainspaceData) => {
        if (currTab === "upload-dataset") {
          uploadDatasetMethods.handleSubmit((data) => {
            setTrainspaceDataset(trainspaceData, data);
          })();
        } else {
          defaultDatasetMethods.handleSubmit((data) => {
            setTrainspaceDataset(trainspaceData, data);
          })();
        }
        dispatch(setTrainspaceData(trainspaceData));
      })}
    />
  );
};

export default TabularDatasetStep;
