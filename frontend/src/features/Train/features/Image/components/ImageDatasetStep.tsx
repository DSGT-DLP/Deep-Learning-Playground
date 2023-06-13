import React, { useEffect } from "react";
import { useForm } from "react-hook-form";
import { DatasetData } from "@/features/Train/types/trainTypes";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import {
    DatasetStepTabLayout,
    DefaultDatasetPanel,
    UploadDatasetPanel,
} from "@/features/Train/components/DatasetStepLayout";
import { TrainspaceData } from "../types/imageTypes";
import { updateImageTrainspaceData } from "../redux/imageActions";

const ImageDatasetStep = ({
    renderStepperButtons,
    setIsModified,
  }: {
    renderStepperButtons: (
      submitTrainspace: (data: TrainspaceData<"DATASET">) => void
    ) => React.ReactNode;
    setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
  }) => {
    const trainspace = useAppSelector(
      (state) => state.trainspace.current as TrainspaceData | undefined
    );
    const [currTab, setCurrTab] = React.useState(
      trainspace?.datasetData?.isDefaultDataset === true
        ? "default-dataset"
        : "upload-dataset"
    );
    const defaultDatasetMethods = useForm<DatasetData>({
      defaultValues:
        trainspace?.datasetData?.isDefaultDataset === true
          ? trainspace?.datasetData
          : undefined,
    });
    const uploadDatasetMethods = useForm<DatasetData>({
      defaultValues:
        trainspace?.datasetData ??
        trainspace?.datasetData?.isDefaultDataset === false
          ? trainspace?.datasetData
          : undefined,
    });
    useEffect(() => {
      setIsModified(
        defaultDatasetMethods.formState.isDirty ||
          uploadDatasetMethods.formState.isDirty
      );
    }, [
      defaultDatasetMethods.formState.isDirty,
      uploadDatasetMethods.formState.isDirty,
    ]);
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
              dataSource={"IMAGE"}
              methods={uploadDatasetMethods}
              acceptedTypes={".zip"}
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
                updateImageTrainspaceData({
                  current: {
                    ...trainspaceData,
                    datasetData: data,
                    parameterData: undefined,
                    reviewData: undefined,
                  },
                  stepLabel: "DATASET",
                })
              );
            })();
          } else {
            defaultDatasetMethods.handleSubmit((data) => {
              dispatch(
                updateImageTrainspaceData({
                  current: {
                    ...trainspaceData,
                    datasetData: data,
                    parameterData: undefined,
                    reviewData: undefined,
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
  
export default ImageDatasetStep;
