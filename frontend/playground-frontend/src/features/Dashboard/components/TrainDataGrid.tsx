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
import { TrainSpaceData } from "../types/trainTypes";
import { IconButton } from "gestalt";
import { Button, Menu, MenuItem, Typography } from "@mui/material";
import { KeyboardArrowDown } from "@mui/icons-material";

const TrainDataGrid = ({
  trainSpaceDataArr,
}: {
  trainSpaceDataArr?: TrainSpaceData[];
}) => {
  return (
    <>
      {trainSpaceDataArr ? (
        <DataGrid
          initialState={{
            sorting: {
              sortModel: [{ field: "timestamp", sort: "desc" }],
            },
            pagination: { paginationModel: { pageSize: 10, page: 0 } },
          }}
          pageSizeOptions={[10]}
          rows={trainSpaceDataArr}
          getRowId={(row) => row.execution_id}
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
              field: "data_source",
              headerName: "Source",
              flex: 1,
              minWidth: 150,
            },
            {
              field: "timestamp",
              headerName: "Date",
              flex: 1,
              minWidth: 150,
              valueFormatter: (params) => formatDate(new Date(params.value)),
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

const formatDate = (date: Date) => {
  const currDate = new Date();

  const time = sameDay(date, currDate)
    ? date.toLocaleTimeString(undefined, {
        hour: "2-digit",
        minute: "2-digit",
      }) + ", "
    : "";

  return (
    time +
    date.toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
      year:
        date.getFullYear() === currDate.getFullYear() ? undefined : "numeric",
    })
  );
};

const sameDay = (d1: Date, d2: Date) => {
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
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
          New Train Space
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
        <MenuItem onClick={handleClose} disableRipple>
          Tabular
        </MenuItem>
        <MenuItem onClick={handleClose} disableRipple>
          Image
        </MenuItem>
        <MenuItem onClick={handleClose} disableRipple>
          Classical ML
        </MenuItem>
        <MenuItem onClick={handleClose} disableRipple>
          Object Detection
        </MenuItem>
      </Menu>
    </div>
  );
};

export default TrainDataGrid;
