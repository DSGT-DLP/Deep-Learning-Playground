import {
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from "@mui/material";
import React from "react";
import PropTypes from "prop-types";
import ArrowForwardIcon from "@mui/icons-material/ArrowForward";
import { useNavigate } from "react-router-dom";

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
  if (statusType === "queued") {
    return (
      <button className="grid-status-display grid-status-display-gray">
        Queued: {status}
      </button>
    );
  } else if (statusType === "training") {
    return (
      <button className="grid-status-display grid-status-display-yellow">
        Training: {status}
      </button>
    );
  } else if (statusType === "finished") {
    return (
      <button className="grid-status-display grid-status-display-green">
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
  return (
    <TableContainer style={{ display: "flex", justifyContent: "center" }}>
      <Table sx={{ minWidth: 400, maxWidth: 1400 }}>
        <TableHead>
          <TableRow>
            <TableCell>Name</TableCell>
            <TableCell>Type</TableCell>
            <TableCell align="right">Input</TableCell>
            <TableCell align="right">Date</TableCell>
            <TableCell align="right">Status</TableCell>
            <TableCell align="right">Result</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.id}
              sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {row.name}
              </TableCell>
              <TableCell component="th" scope="row">
                {row.type}
              </TableCell>
              <TableCell align="right">{row.input}</TableCell>
              <TableCell align="right">{formatDate(row.date)}</TableCell>
              <TableCell align="right">
                <StatusDisplay
                  statusType={row.statusType}
                  status={row.status}
                />
              </TableCell>
              <TableCell align="right">
                <button className="grid-status-display grid-status-display-blue" >
                  RESULT
                </button>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );
};

const Dashboard = () => {
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
