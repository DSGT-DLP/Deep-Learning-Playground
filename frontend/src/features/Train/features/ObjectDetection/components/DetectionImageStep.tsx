import FilerobotImageEditor, {
  TABS,
  TOOLS,
} from "react-filerobot-image-editor";
import { UploadImagePanel } from "@/features/Train/components/ImageUploadLayout";
import React, { useEffect } from "react";
import { TrainspaceData } from "../types/detectionTypes";
import { useLazySendUploadDataQuery } from "../redux/uploadApi";
import { Stack } from "@mui/material";
import { updateDetectionTrainspaceData } from "../redux/detectionActions";
import { useForm, Controller } from "react-hook-form";
import { useAppDispatch, useAppSelector } from "@/common/redux/hooks";
import { ImageUploadData } from "@/features/Train/types/trainTypes";
import {
  useGetDatasetFilesDataQuery,
  useUploadDatasetFileMutation,
} from "@/features/Train/redux/trainspaceApi";

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
  const {
    handleSubmit,
    formState: { errors, isDirty },
    control,
  } = useForm<ImageData>();
  const [uploadFile] = useUploadDatasetFileMutation();
  const uploadImageMethods = useForm<ImageUploadData>();
  const dispatch = useAppDispatch();
  const dataURLtoFile = (dataurl: string, filename: string) => {
    const arr = dataurl.split(",");
    if (arr.length === 0) {
      return new File([""], filename);
    }
    const matched = arr[0].match(/:(.*?);/);
    const mime = matched ? matched[1] : undefined;
    const bstr = atob(arr[1]);
    let n = bstr.length;
    const u8arr = new Uint8Array(n);
    while (n) {
      u8arr[n - 1] = bstr.charCodeAt(n - 1);
      n -= 1; // to make eslint happy
    }
    return new File([u8arr], filename, { type: mime });
  };
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
                imageData: data,
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
