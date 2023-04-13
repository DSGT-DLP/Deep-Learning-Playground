import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button as MuiButton,
  Typography,
  Menu,
  MenuItem,
} from "@mui/material";
import {
  Box,
  Button,
  Flex,
  IconButton,
  Dropdown,
  PageHeader,
  Spinner,
} from "gestalt";
import "gestalt/dist/gestalt.css";
import { useEffect, useState } from "react";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import JSZip from "jszip";
import saveAs from "file-saver";
import { Doughnut, Bar } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  LinearScale,
  BarElement,
  Title,
  TimeSeriesScale,
  ChartData,
} from "chart.js";
import "chartjs-adapter-date-fns";
import { enUS } from "date-fns/locale";
import { format, isFuture, add } from "date-fns";
import React from "react";
import { useGetExecutionsDataQuery } from "@/common/redux/backendApi";
import NavbarMain from "@/common/components/NavBarMain";
import Footer from "@/common/components/Footer";
import { useAppSelector } from "@/common/redux/hooks";
import { useRouter } from "next/router";
import { isSignedIn } from "@/common/redux/userLogin";
import { AgGridReact } from "ag-grid-react";
import "ag-grid-community/styles/ag-grid.css";
import "ag-grid-community/styles/ag-theme-alpine.css";
import {
  DataGrid,
  GridAddIcon,
  GridColDef,
  GridDeleteIcon,
  GridToolbar,
  GridToolbarContainer,
  GridToolbarDensitySelector,
  GridToolbarExport,
  GridToolbarFilterButton,
} from "@mui/x-data-grid";
import { ArrowDropDown, KeyboardArrowDown, Margin } from "@mui/icons-material";
import { TrainSpaceData } from "@/features/Dashboard/types/train_types";

ChartJS.register(
  ArcElement,
  Tooltip,
  Legend,
  LinearScale,
  BarElement,
  Title,
  TimeSeriesScale
);

const BlankGrid = () => {
  return (
    <div id="blank-grid-wrapper">
      <div id="blank-grid">
        <p>
          You haven't trained any models yet. Create your first model below!
        </p>
        <button
          id="blank-grid-button"
          onClick={() => console.log("To be implemented")}
        >
          Train Model
        </button>
      </div>
    </div>
  );
};

const StatusDisplay = ({ statusType }: { statusType: string }) => {
  if (statusType === "QUEUED") {
    return (
      <button
        className="grid-status-display grid-status-display-gray"
        onClick={() => console.log("To be implemented")}
      >
        Queued
      </button>
    );
  } else if (statusType === "STARTING") {
    return (
      <button
        className="grid-status-display grid-status-display-yellow"
        onClick={() => console.log("To be implemented")}
      >
        Training...
      </button>
    );
  } else if (statusType === "UPLOADING") {
    return (
      <button
        className="grid-status-display grid-status-display-blue"
        onClick={() => console.log("To be implemented")}
      >
        Uploading...
      </button>
    );
  } else if (statusType === "TRAINING") {
    return (
      <button
        className="grid-status-display grid-status-display-blue"
        onClick={() => console.log("To be implemented")}
      >
        Training...
      </button>
    );
  } else if (statusType === "ERROR") {
    return (
      <button
        className="grid-status-display grid-status-display-red"
        onClick={() => console.log("To be implemented")}
      >
        Error
      </button>
    );
  } else if (statusType === "SUCCESS") {
    return (
      <button
        className="grid-status-display grid-status-display-green"
        onClick={() => console.log("To be implemented")}
      >
        Done <ArrowForwardIcon fontSize="small" />
      </button>
    );
  } else {
    return <p>Incorrect status type passed</p>;
  }
};

const sameDay = (d1: Date, d2: Date) => {
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
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

const FilledGrid = (props: { executionTable: TrainSpaceData[] }) => {
  const { executionTable } = props;
  function toTitleCase(str: string) {
    return str.replace(/\w\S*/g, function (txt) {
      return txt.charAt(0).toUpperCase() + txt.substring(1).toLowerCase();
    });
  }
  async function handleOnDownloadClick(e: unknown, row: TrainSpaceData) {
    (e as Event).stopPropagation();
    /*
    const response = await sendToBackend("getExecutionsFilesPresignedUrls", {
      exec_id: row.execution_id,
    });
    const zip = new JSZip();
    await Promise.all(
      [
        [response.dl_results, "dl_results.csv"],
        [response.model_onnx, "my_deep_learning_model.onnx"],
        [response.model_pt, "model.pt"],
      ].map(([url, filename]) =>
        fetch(url, {
          mode: "cors",
        }).then((res) =>
          res.blob().then((blob) => {
            zip.file(filename, blob);
          })
        )
      )
    );
    zip
      .generateAsync({ type: "blob" })
      .then((blob) => saveAs(blob, "results.zip"));*/
  }
  return (
    <>
      {executionTable ? (
        <TableContainer style={{ display: "flex", justifyContent: "center" }}>
          <Table sx={{ minWidth: 400, m: 2 }}>
            <TableHead>
              <TableRow>
                <TableCell className="dashboard-header">Name</TableCell>
                <TableCell className="dashboard-header">Type</TableCell>
                <TableCell className="dashboard-header" align="left">
                  Date
                </TableCell>
                <TableCell className="dashboard-header" align="left">
                  Status
                </TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {executionTable.map((row) => (
                <TableRow
                  key={row.execution_id}
                  sx={{
                    "&:last-child td, &:last-child th": { border: 0 },
                    cursor: "pointer",
                  }}
                  onClick={() => console.log("To be implemented")}
                  hover
                >
                  <TableCell
                    component="th"
                    scope="row"
                    className="dashboard-header"
                  >
                    {row.name}
                  </TableCell>
                  <TableCell component="th" scope="row" className="row-style">
                    {toTitleCase(row.data_source)}
                  </TableCell>
                  <TableCell align="left" className="row-style">
                    {formatDate(new Date(row.timestamp))}
                  </TableCell>
                  <TableCell align="left">
                    <StatusDisplay statusType={row.status} />
                  </TableCell>
                  <TableCell align="left">
                    <IconButton
                      icon="download"
                      accessibilityLabel={"Download"}
                      size={"md"}
                      disabled={row.status !== "SUCCESS"}
                      onClick={(e) => handleOnDownloadClick(e.event, row)}
                    ></IconButton>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TableContainer>
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
        <MuiButton
          variant="contained"
          startIcon={<GridAddIcon></GridAddIcon>}
          style={{ margin: "10px" }}
          endIcon={<KeyboardArrowDown></KeyboardArrowDown>}
          onClick={(e) => setAnchorEl(e.currentTarget)}
        >
          New Train Space
        </MuiButton>
        <NewTrainSpaceMenu anchorEl={anchorEl} setAnchorEl={setAnchorEl} />
        <GridToolbarFilterButton />
        <GridToolbarDensitySelector />
        <GridToolbarExport />
        <MuiButton variant="text" startIcon={<GridDeleteIcon></GridDeleteIcon>}>
          Delete
        </MuiButton>
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

const Dashboard = () => {
  const [modelTypeDoughnutData, setModelTypeDoughnutData] =
    useState<ChartData<"doughnut"> | null>(null);
  const [execFrequencyBarData, setExecFrequencyBarData] = useState<ChartData<
    "bar",
    { x: Date; y: number }[]
  > | null>(null);
  const {
    data: executionTable,
    isLoading: executionTableLoading,
    refetch: refetchExecutionTable,
  } = useGetExecutionsDataQuery();

  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  useEffect(() => {
    if (!user) {
      router.replace("/login");
    }
  }, [user]);
  useEffect(() => {
    if (executionTable) {
      setModelTypeDoughnutData({
        datasets: [
          {
            data: [
              executionTable.filter((row) => row.data_source === "TABULAR")
                .length,
              executionTable.filter((row) => row.data_source === "IMAGE")
                .length,
            ],
            backgroundColor: [
              "rgb(255, 99, 132)",
              "rgb(54, 162, 235)",
              "rgb(255, 205, 86)",
            ],
            label: "Frequency",
          },
        ],
        labels: ["Tabular", "Image"],
      });
      setModelTypeDoughnutData({
        datasets: [
          {
            data: [
              executionTable.filter((row) => row.data_source === "TABULAR")
                .length,
              executionTable.filter((row) => row.data_source === "IMAGE")
                .length,
            ],
            backgroundColor: [
              "rgb(255, 99, 132)",
              "rgb(54, 162, 235)",
              "rgb(255, 205, 86)",
            ],
            label: "Frequency",
          },
        ],
        labels: ["Tabular", "Image"],
      });
      const sameDay = (d1: Date, d2: Date) => {
        return (
          d1.getFullYear() === d2.getFullYear() &&
          d1.getMonth() === d2.getMonth() &&
          d1.getDate() === d2.getDate()
        );
      };
      const setToNearestDay = (d: Date) => {
        d.setHours(0, 0, 0, 0);
        return d;
      };
      const execFrequencyData: { x: Date; y: number }[] = [];
      executionTable.forEach((row) => {
        if (isFuture(add(new Date(row.timestamp), { days: 30 }))) {
          execFrequencyData.length !== 0 &&
          sameDay(
            new Date(row.timestamp),
            execFrequencyData[execFrequencyData.length - 1].x
          )
            ? (execFrequencyData[execFrequencyData.length - 1].y += 1)
            : execFrequencyData.push({
                x: setToNearestDay(new Date(row.timestamp)),
                y: 1,
              });
        }
      });
      setExecFrequencyBarData({
        datasets: [
          {
            label: "# of Executions",
            backgroundColor: "rgba(75, 192, 192, 0.7)",
            borderColor: "rgb(75, 192, 192)",
            borderWidth: 1,
            barThickness: 15,
            data: execFrequencyData,
          },
        ],
      });
    }
  }, [executionTable]);
  if (!isSignedIn(user)) {
    return <></>;
  }
  return (
    <>
      <NavbarMain />
      <div id="dashboard">
        <>
          <PageHeader
            maxWidth="85%"
            title="Dashboard"
            primaryAction={{
              component: (
                <Button
                  color="red"
                  size="md"
                  iconEnd="refresh"
                  text="Refresh"
                  onClick={() => {
                    refetchExecutionTable();
                  }}
                />
              ),
              dropdownItems: [
                <Dropdown.Item
                  key="refresh"
                  option={{ value: "refresh", label: "Refresh" }}
                  onSelect={() => {
                    refetchExecutionTable();
                  }}
                />,
              ],
            }}
            dropdownAccessibilityLabel="More options"
          />
          <Flex
            direction="row"
            justifyContent="center"
            alignItems="stretch"
            width="100%"
            wrap
          >
            {modelTypeDoughnutData ? (
              <Box>
                <Doughnut data={modelTypeDoughnutData} />
              </Box>
            ) : null}
            {execFrequencyBarData ? (
              <Box height={300} width={300}>
                <Bar
                  data={execFrequencyBarData}
                  options={{
                    maintainAspectRatio: false,
                    scales: {
                      x: {
                        adapters: {
                          date: {
                            locale: enUS,
                          },
                        },
                        ticks: {
                          maxRotation: 80,
                          minRotation: 80,
                        },
                        type: "timeseries",
                        time: {
                          unit: "day",
                          minUnit: "day",
                          displayFormats: {
                            day: "MMM dd",
                          },
                        },
                      },
                      y: {
                        beginAtZero: true,
                      },
                    },
                    responsive: true,
                    plugins: {
                      tooltip: {
                        callbacks: {
                          title: (context) => {
                            return format(
                              execFrequencyBarData.datasets[0].data[
                                context[0].dataIndex
                              ].x,
                              "MMM d"
                            );
                          },
                        },
                      },
                      legend: {
                        display: false,
                      },
                      title: {
                        display: true,
                        text: "Training Frequency",
                      },
                    },
                  }}
                />
              </Box>
            ) : null}
          </Flex>
          {executionTable && (
            <div
              style={{
                minWidth: "900px",
                width: "75%",
                margin: "auto",
              }}
            >
              <DataGrid
                initialState={{
                  sorting: {
                    sortModel: [{ field: "timestamp", sort: "desc" }],
                  },
                  pagination: { paginationModel: { pageSize: 10, page: 0 } },
                }}
                pageSizeOptions={[10]}
                rows={executionTable}
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
                    renderHeader: (_) => {
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
                    valueFormatter: (params) =>
                      formatDate(new Date(params.value)),
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
            </div>
          )}
          {executionTable && executionTable.length === 0 && <BlankGrid />}
          {executionTableLoading ? (
            <div className="loading">
              <Spinner show accessibilityLabel="Spinner" />
            </div>
          ) : null}
        </>
      </div>
      <Footer />
    </>
  );
};

export default Dashboard;
