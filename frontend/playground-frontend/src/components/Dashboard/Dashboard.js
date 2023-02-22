import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import { Box, Button } from "gestalt";
import "gestalt/dist/gestalt.css";
import React, { useEffect, useState } from "react";
import PropTypes from "prop-types";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import { sendToBackend } from "../helper_functions/TalkWithBackend";
import "./../../App.css";
import { auth } from "../../firebase";

const rows = [
  {
    id: "123mrpij",
    name: "IrisDense",
    type: "Tabular",
    input: "new_input.csv",
    statusType: "queued",
    status: "1 / 4",
    date: 1662850862,
  },
  {
    id: "as98dfumasdp",
    name: "Penguin",
    type: "Tabular",
    input: "my_tabular_input.csv",
    statusType: "training",
    status: "35%",
    date: 1662750862,
  },
  {
    id: "p9umaspdf",
    name: "Iris",
    type: "Image Training",
    input: "my_images.zip",
    statusType: "finished",
    status: "Done",
    date: 1441850862,
  },
];

const BlankGrid = () => {
  const navigate = useNavigate();

  return (
    <div id="blank-grid-wrapper">
      <div id="blank-grid">
        <p>
          You haven't trained any models yet. Create your first model below!
        </p>
        <button id="blank-grid-button" onClick={() => navigate("/")}>
          Train Model
        </button>
      </div>
    </div>
  );
};

const StatusDisplay = ({ statusType, status }) => {
  const navigate = useNavigate();
  if (statusType === "queued") {
    return (
      <button
        className="grid-status-display grid-status-display-gray"
        onClick={() => navigate("/")}
      >
        Queued: {status}
      </button>
    );
  } else if (statusType === "STARTING") {
    return (
      <button
        className="grid-status-display grid-status-display-yellow"
        onClick={() => navigate("/")}
      >
        Training: {status}
      </button>
    );
  } else if (statusType === "SUCCESS") {
    return (
      <button
        className="grid-status-display grid-status-display-green"
        onClick={() => navigate("/")}
      >
        Done <ArrowForwardIcon fontSize="small" />
      </button>
    );
  } else {
    return <p>Incorrect status type passed</p>;
  }
};

const sameDay = (d1, d2) => {
  return (
    d1.getFullYear() === d2.getFullYear() &&
    d1.getMonth() === d2.getMonth() &&
    d1.getDate() === d2.getDate()
  );
};

const formatDate = (date) => {
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

const FilledGrid = () => {
  const navigate = useNavigate();
  const [executiontable, setUserExecutionTable] = useState(null);
  useEffect(() => {
    getExecutionTable();
  }, [auth.currentUser]);
  const getExecutionTable = async () => {
    if (auth.currentUser) {
      const response = await sendToBackend("executiontable", {});
      setUserExecutionTable(JSON.parse(response["record"]));
    } else {
      setUserExecutionTable(null);
    }
  };
  function toTitleCase(str) {
    return str.replace(/\w\S*/g, function (txt) {
      return txt.charAt(0).toUpperCase() + txt.substr(1).toLowerCase();
    });
  }
  return (
    <>
      {executiontable ? (
        <>
          <Box padding={5}>
            <Button
              color="red"
              size="lg"
              text="Refresh"
              onClick={() => {
                getExecutionTable();
              }}
            />
          </Box>
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
                {executiontable.map((row) => (
                  <TableRow
                    key={row.execution_id}
                    sx={{
                      "&:last-child td, &:last-child th": { border: 0 },
                      cursor: "pointer",
                    }}
                    onClick={() => navigate("/")}
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
                      <StatusDisplay
                        statusType={row.status}
                        status={`${row.progress.toFixed(2)}%`}
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </>
      ) : null}
    </>
  );
};

const Dashboard = () => {
  const signedInUserEmail = useSelector((state) => state.currentUser.email);
  const navigate = useNavigate();
  useEffect(() => {
    if (signedInUserEmail) navigate("/dashboard");
  }, [signedInUserEmail]);

  return (
    <div id="dashboard">
      <FilledGrid />
      {rows.length === 0 && <BlankGrid />}
    </div>
  );
};

export default Dashboard;

StatusDisplay.propTypes = {
  statusType: PropTypes.oneOf(["queued", "STARTING", "SUCCESS"]),
  status: PropTypes.string,
};
