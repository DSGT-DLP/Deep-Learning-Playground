import { Box, Button, Flex, Dropdown, PageHeader, Spinner } from "gestalt";
import "gestalt/dist/gestalt.css";
import { useEffect } from "react";
//import JSZip from "jszip";
//import saveAs from "file-saver";
import React from "react";
import NavbarMain from "@/common/components/NavBarMain";
import Footer from "@/common/components/Footer";
import { useAppSelector } from "@/common/redux/hooks";
import { useRouter } from "next/router";
import { isSignedIn } from "@/common/redux/userLogin";
import { useGetExecutionsDataQuery } from "@/features/Dashboard/redux/dashboardApi";
import TrainDoughnutChart from "@/features/Dashboard/components/TrainDoughnutChart";
import TrainBarChart from "@/features/Dashboard/components/TrainBarChart";
import TrainDataGrid from "@/features/Dashboard/components/TrainDataGrid";

const Dashboard = () => {
  const { data, isLoading, refetch } = useGetExecutionsDataQuery();
  const user = useAppSelector((state) => state.currentUser.user);
  const router = useRouter();
  useEffect(() => {
    if (!user) {
      router.replace("/login");
    }
  }, [user]);
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
                    refetch();
                  }}
                />
              ),
              dropdownItems: [
                <Dropdown.Item
                  key="refresh"
                  option={{ value: "refresh", label: "Refresh" }}
                  onSelect={() => {
                    refetch();
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
            <Box>
              <TrainDoughnutChart trainSpaceDataArr={data} />
            </Box>
            <Box height={300} width={300}>
              <TrainBarChart trainSpaceDataArr={data} />
            </Box>
          </Flex>

          <div
            style={{
              minWidth: "900px",
              width: "75%",
              margin: "auto",
            }}
          >
            <TrainDataGrid trainSpaceDataArr={data} />
          </div>
          {data && data.length === 0 && <BlankGrid />}
          {isLoading ? (
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

/*
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
};*/

/*
const FilledGrid = (props: { executionTable: TrainSpaceData[] }) => {
  const { executionTable } = props;
  function toTitleCase(str: string) {*/
//return str.replace(/\w\S*/g, function (txt) {
//  return txt.charAt(0).toUpperCase() + txt.substring(1).toLowerCase();
//});
/*}
  async function handleOnDownloadClick(e: unknown, row: TrainSpaceData) {
    (e as Event).stopPropagation();
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
      .then((blob) => saveAs(blob, "results.zip"));
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
};*/

export default Dashboard;
