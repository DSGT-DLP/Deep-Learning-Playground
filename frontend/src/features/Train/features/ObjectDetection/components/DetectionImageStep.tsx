import { UploadImagePanel } from "@/features/Train/components/ImageUploadLayout";
import React, { useEffect } from "react";
import { TrainspaceData } from "../types/detectionTypes";
import { Stack } from "@mui/material";
import { updateDetectionTrainspaceData } from "../redux/detectionActions";
import { useForm } from "react-hook-form";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { ImageUploadData } from "@/features/Train/types/trainTypes";

const DetectionImageStep = ({
  renderStepperButtons,
  setIsModified,
}: {
  renderStepperButtons: (
    submitTrainspace: (data: TrainspaceData<"IMAGE">) => void
  ) => React.ReactNode;
  setIsModified: React.Dispatch<React.SetStateAction<boolean>>;
}) => {
  const trainspace = useAppSelector(
    (state) => state.trainspace.current as TrainspaceData | undefined
  );
  const uploadImageMethods = useForm<ImageUploadData>({
    defaultValues: trainspace?.imageData,
  });
  useEffect(() => {
    setIsModified(uploadImageMethods.formState.isDirty);
  }, [uploadImageMethods.formState.isDirty]);
  const dispatch = useAppDispatch();
  return (
    <Stack spacing={3}>
      <UploadImagePanel
        dataSource={"OBJECT_DETECTION"}
        methods={uploadImageMethods}
      />
      {renderStepperButtons((trainspaceData) => {
        uploadImageMethods.handleSubmit((data) => {
          dispatch(
            updateDetectionTrainspaceData({
              current: {
                ...trainspaceData,
                datasetData: data,
              },
              stepLabel: "IMAGE",
            })
          );
        })();
      })}
    </Stack>
  );
};

export default DetectionImageStep;
