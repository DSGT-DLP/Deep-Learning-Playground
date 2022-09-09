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
    type: "Tabular",
    input: "new_input.csv",
    statusType: "queued",
    status: "1 / 4",
    date: "Sep 5, 2022",
  },
  {
    id: "as98dfumasdp",
    type: "Tabular",
    input: "my_tabular_input.csv",
    statusType: "training",
    status: "35%",
    date: "Sep 4, 2022",
  },
  {
    id: "p9umaspdf",
    type: "Image Training",
    input: "my_images.zip",
    statusType: "finished",
    status: "Done",
    date: "Sep 1, 2022",
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

const FilledGrid = () => {
  return (
    <TableContainer>
      <Table sx={{ minWidth: 400 }}>
        <TableHead>
          <TableRow>
            <TableCell>Type</TableCell>
            <TableCell align="right">Input</TableCell>
            <TableCell align="right">Date</TableCell>
            <TableCell align="right">Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((row) => (
            <TableRow
              key={row.id}
              sx={{ "&:last-child td, &:last-child th": { border: 0 } }}
            >
              <TableCell component="th" scope="row">
                {row.type}
              </TableCell>
              <TableCell align="right">{row.input}</TableCell>
              <TableCell align="right">{row.date}</TableCell>
              <TableCell align="right">
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
