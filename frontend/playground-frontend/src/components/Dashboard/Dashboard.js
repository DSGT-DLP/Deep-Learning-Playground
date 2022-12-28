import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import React, { useEffect } from "react";
import PropTypes from "prop-types";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { useNavigate } from "react-router-dom";
import { useSelector } from "react-redux";
import "./../../App.css";

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
  } else if (statusType === "training") {
    return (
      <button
        className="grid-status-display grid-status-display-yellow"
        onClick={() => navigate("/")}
      >
        Training: {status}
      </button>
    );
  } else if (statusType === "finished") {
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

const formatDate = (unixTime) => {
  const currDate = new Date();
  const date = new Date(unixTime * 1000);

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

  return (
    <TableContainer style={{ display: "flex", justifyContent: "center" }}>
      <Table sx={{ minWidth: 400, m: 2 }}>
        <TableHead>
          <TableRow>
            <TableCell className="dashboard-header">Name</TableCell>
            <TableCell className="dashboard-header">Type</TableCell>
            <TableCell className="dashboard-header" align="left">
              Input
            </TableCell>
            <TableCell className="dashboard-header" align="left">
              Date
            </TableCell>
            <TableCell className="dashboard-header" align="left">
              Status
            </TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.id}
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
                {row.type}
              </TableCell>
              <TableCell align="left" className="row-style">
                {row.input}
              </TableCell>
              <TableCell align="left" className="row-style">
                {formatDate(row.date)}
              </TableCell>
              <TableCell align="left">
                <StatusDisplay
                  statusType={row.statusType}
                  status={row.status}
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
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
  statusType: PropTypes.oneOf(["queued", "training", "finished"]),
  status: PropTypes.string,
};
