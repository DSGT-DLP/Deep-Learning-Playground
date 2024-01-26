import {
  DataGrid,
  GridAddIcon,
  GridDeleteIcon,
  GridToolbarContainer,
  GridToolbarDensitySelector,
  GridToolbarExport,
  GridToolbarFilterButton,
} from "@mui/x-data-grid";
import React, { useEffect, useState } from "react";
import { TrainResultsData } from "@/features/Train/types/trainTypes";
import { IconButton } from "gestalt";
import { Button, Menu, MenuItem, Typography } from "@mui/material";
import { KeyboardArrowDown } from "@mui/icons-material";
import { useRouter } from "next/router";
import { useAppDispatch } from "@/common/redux/hooks";
import {
  ALL_TRAINSPACE_SETTINGS,
  IMPLEMENTED_DATA_SOURCE_ARR,
} from "@/features/Train/constants/trainConstants";
import { formatDate } from "@/common/utils/dateFormat";
import { removeTrainspaceData } from "@/features/Train/redux/trainspaceSlice";

const TrainDataGrid = ({
  trainSpaceDataArr,
}: {
  trainSpaceDataArr?: TrainResultsData[];
}) => {
  return (
    <>
      {trainSpaceDataArr ? (
        <DataGrid
          initialState={{
            sorting: {
              sortModel: [{ field: "created", sort: "desc" }],
            },
            pagination: { paginationModel: { pageSize: 10, page: 0 } },
          }}
          pageSizeOptions={[10]}
          rows={trainSpaceDataArr}
          getRowId={(row) => row.trainspaceId}
          autoHeight
          disableColumnMenu
          slots={{
            toolbar: CustomGridToolBar,
          }}
          density="comfortable"
          columns={[
            {
              field: "train",
              width: 75,
              filterable: false,
              sortable: false,
              hideable: false,
              disableColumnMenu: true,
              renderHeader: () => {
                return (
                  <div style={{ width: "45px", textAlign: "center" }}>
                    Train
                  </div>
                );
              },
              renderCell: (params) => {
                const rowElement = params.api.getRowElement(params.id);
                if (rowElement) {
                  return <GridPlayButton rowElement={rowElement} />;
                }
              },
            },
            { field: "name", headerName: "Name", flex: 2, minWidth: 300 },
            {
              field: "dataSource",
              headerName: "Source",
              flex: 1,
              minWidth: 150,
            },
            {
              field: "created",
              headerName: "Date",
              flex: 1,
              minWidth: 150,
              valueFormatter: (params) => formatDate(params.value),
            },
            {
              field: "status",
              headerName: "Status",
              flex: 1,
              minWidth: 150,
            },
          ]}
          checkboxSelection
        />
      ) : null}
    </>
  );
};

const GridPlayButton = ({ rowElement }: { rowElement: HTMLDivElement }) => {
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    rowElement.onmouseenter = () => setVisible(true);
    rowElement.onmouseleave = () => setVisible(false);
  }, []);
  return (
    <>
      {visible ? (
        <IconButton
          accessibilityLabel="Open"
          icon="play"
          onClick={(e) => {
            e.event.stopPropagation();
            console.log("To be implemented");
          }}
        />
      ) : null}
    </>
  );
};

const CustomGridToolBar = () => {
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  return (
    <>
      <Typography variant="h2" fontSize={25} margin={2}>
        Train Spaces
      </Typography>

      <GridToolbarContainer>
        <Button
          variant="contained"
          startIcon={<GridAddIcon></GridAddIcon>}
          style={{ margin: "10px" }}
          endIcon={<KeyboardArrowDown></KeyboardArrowDown>}
          onClick={(e) => setAnchorEl(e.currentTarget)}
        >
          Create Model
        </Button>
        <NewTrainSpaceMenu anchorEl={anchorEl} setAnchorEl={setAnchorEl} />
        <GridToolbarFilterButton />
        <GridToolbarDensitySelector />
        <GridToolbarExport />
        <Button variant="text" startIcon={<GridDeleteIcon></GridDeleteIcon>}>
          Delete
        </Button>
      </GridToolbarContainer>
    </>
  );
};

const NewTrainSpaceMenu = ({
  anchorEl,
  setAnchorEl,
}: {
  anchorEl: null | HTMLElement;
  setAnchorEl: React.Dispatch<React.SetStateAction<null | HTMLElement>>;
}) => {
  const open = Boolean(anchorEl);
  const handleClose = () => {
    setAnchorEl(null);
  };
  const router = useRouter();
  const dispatch = useAppDispatch();
  return (
    <div>
      <Menu
        MenuListProps={{
          "aria-labelledby": "demo-customized-button",
        }}
        anchorOrigin={{
          vertical: "bottom",
          horizontal: "right",
        }}
        transformOrigin={{
          vertical: "top",
          horizontal: "right",
        }}
        anchorEl={anchorEl}
        open={open}
        onClose={handleClose}
      >
        {IMPLEMENTED_DATA_SOURCE_ARR.map((source) => (
          <MenuItem
            key={source}
            value={source}
            onClick={() => {
              handleClose();
              dispatch(removeTrainspaceData());
              router.push({ pathname: "/train", query: { source } });
            }}
          >
            {ALL_TRAINSPACE_SETTINGS[source].name}
          </MenuItem>
        ))}
      </Menu>
    </div>
  );
};

export default TrainDataGrid;
